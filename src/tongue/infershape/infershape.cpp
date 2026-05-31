#include "infershape.h"
#include "builder/ops/ops.h"
#include "core/config.h"
#include "core/graph.h"
#include "core/subgraph.h"
#include "utils/utils.h"

namespace tongue
{
using namespace aipubt;

std::map<OpType, InferFunc> &get_infer_map(bool backward)
{
    static std::map<OpType, InferFunc> forward_map, backward_map;
    if (backward)
    {
        return backward_map;
    }
    return forward_map;
}

std::vector<SymbolShape> infer_shape(NodePtr n, const std::vector<SymbolShape> &inputs, bool backward)
{
    auto &infer_map = get_infer_map(backward);
    if (infer_map.count(n->type()))
    {
        return infer_map[n->type()](n, inputs);
    }
    return std::vector<SymbolShape>();
}

static void handle_backward_layout(NodePtr n, std::vector<SymbolShape> &shapes)
{
    auto inputs = n->inputs();
    if (inputs.size() != shapes.size())
    {
        return;
    }
    for (uint32_t i = 0; i < inputs.size(); ++i)
    {
        auto s = inputs[i]->shape();
        auto layout = s.layout();
        if (layout == DataLayout_NHWC)
        {
            continue;
        }
        s.change_layout(DataLayout_NHWC);
        s.symbol() = SymbolShape(shapes[i], s.layout(), s.indexing());
        s.change_layout(layout);
        shapes[i] = s.symbol();
    }
}

static bool auto_infer_shape(NodePtr n, const std::vector<SymbolShape> &inputs, std::vector<SymbolShape> &res)
{
    if (!n->params().count("ds_output_shape"))
    {
        return false;
    }

    std::vector<std::vector<Expr>> ds_out_shapes = n->params()["ds_output_shape"];
    if (n->outputs().size() != ds_out_shapes.size())
    {
        return false;
    }

    for (size_t i = 0; i < n->outputs().size(); ++i)
    {
        if (n->outputs()[i]->shape().dim() != (int)ds_out_shapes[i].size())
        {
            return false;
        }
    }

    res.clear();
    std::vector<Expr> input_dims;
    for (std::vector<Expr> s : inputs)
    {
        input_dims.insert(input_dims.end(), s.begin(), s.end());
    }

    for (size_t i = 0; i < n->outputs().size(); ++i)
    {
        std::vector<Expr> s;
        std::map<std::string, ExprPtr> symbols;
        for (auto &&expr : ds_out_shapes[i])
        {
            for (auto &var : expr->variables())
            {
                auto name = var.name();
                if (name.size() > 1 && utils::startswith(name, "s"))
                {
                    auto str = name.substr(1, -1);
                    if (std::all_of(str.begin(), str.end(), ::isdigit))
                    {
                        int idx = std::stoi(str);
                        if (idx >= 0 && idx < (int)input_dims.size())
                        {
                            symbols[name] = input_dims[idx];
                        }
                    }
                }
            }
            s.push_back(ExprImpl::replace_symbol(expr, symbols));
        }
        res.push_back(SymbolShape(s));
    }

    return true;
}

static std::vector<SymbolShape> infer_eltwise(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1 && inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() == 2 && inputs[0].dim() != inputs[1].dim())
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].is_constant() && !inputs[1].is_constant())
    {
        return {inputs[1]};
    }

    return {inputs[0]};
}

static std::vector<SymbolShape> infer_eltwise_backward(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    return {inputs[0], inputs[0]};
}

static std::vector<SymbolShape> infer_local(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    auto &params = n->params();
    if (!params.count("kernel_x") || !params.count("kernel_y") ||
        !params.count("stride_x") || !params.count("stride_y"))
    {
        return std::vector<SymbolShape>();
    }

    int kernel_x = params["kernel_x"];
    int kernel_y = params["kernel_y"];
    int stride_x = params["stride_x"];
    int stride_y = params["stride_y"];
    int dilation_x = params["dilation_x"];
    int dilation_y = params["dilation_y"];
    int pad_left = params["pad_left"];
    int pad_right = params["pad_right"];
    int pad_top = params["pad_top"];
    int pad_bottom = params["pad_bottom"];
    int out_pad_x = params.count("output_padding_x") ? (int)params["output_padding_x"] : 0;
    int out_pad_y = params.count("output_padding_y") ? (int)params["output_padding_y"] : 0;

    auto i_n = inputs[0].n();
    auto i_h = inputs[0].h();
    auto i_w = inputs[0].w();

    Expr o_c;
    bool is_conv_transpose = n->type() == OpType_ConvTranspose;
    if (n->type() == OpType_Convolution || n->type() == OpType_DepthwiseConv || is_conv_transpose)
    {
        o_c = (int)params["num_output"];
    }
    else if (n->type() == OpType_Pooling || n->type() == OpType_Dilation ||
             n->type() == OpType_Erosion || n->type() == OpType_MaxPoolingWithArgMax)
    {
        o_c = inputs[0].c();
    }

    bool is_ceil_mode = false;
    if (n->type() == OpType_Pooling || n->type() == OpType_MaxPoolingWithArgMax)
    {
        is_ceil_mode = params.count("ceil_mode") ? (bool)params["ceil_mode"] : false;
    }

    auto infer_hw = [is_ceil_mode, is_conv_transpose](Expr in, Expr k, Expr s, Expr d, Expr p0, Expr p1, Expr o_p) -> Expr {
        auto new_k = k + (k - 1) * (d - 1);
        Expr out;
        if (is_conv_transpose)
        {
            out = s * (in - 1) + new_k - (p0 + p1) + o_p;
        }
        else if (is_ceil_mode)
        {
            // out = ((in - new_k + p0 + p1) / s).ceil() + 1;
            out = (in - new_k + p0 + p1).floordiv(s) + 1;
            auto p = Expr::select(((in - new_k + p0 + p1) % s) > 0, 1, 0);
            p = Expr::select((out * s + 1) <= (in + p0 + p1), p, 0);
            out = out + p;
        }
        else
        {
            out = (in - new_k + p0 + p1).floordiv(s) + 1;
        }
        return out;
    };
    auto oh = infer_hw(i_h, kernel_y, stride_y, dilation_y, pad_top, pad_bottom, out_pad_y);
    auto ow = infer_hw(i_w, kernel_x, stride_x, dilation_x, pad_left, pad_right, out_pad_x);

    SymbolShape s({i_n, oh, ow, o_c});
    std::vector<SymbolShape> ret = {s};

    if (n->type() == OpType_MaxPoolingWithArgMax)
    {
        ret.push_back(s);
    }
    return ret;
}

static std::vector<SymbolShape> infer_pooling3d(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    auto &params = n->params();
    if (!params.count("kernel_x") || !params.count("kernel_y") || !params.count("kernel_z") ||
        !params.count("stride_x") || !params.count("stride_y") || !params.count("stride_z"))
    {
        return std::vector<SymbolShape>();
    }

    int kernel_x = params["kernel_x"];
    int kernel_y = params["kernel_y"];
    int kernel_z = params["kernel_z"];
    int stride_x = params["stride_x"];
    int stride_y = params["stride_y"];
    int stride_z = params["stride_z"];
    int dilation_x = params["dilation_x"];
    int dilation_y = params["dilation_y"];
    int dilation_z = params["dilation_z"];
    int pad_z_begin = params["pad_z_begin"];
    int pad_z_end = params["pad_z_end"];
    int pad_y_begin = params["pad_y_begin"];
    int pad_y_end = params["pad_y_end"];
    int pad_x_begin = params["pad_x_begin"];
    int pad_x_end = params["pad_x_end"];

    auto i_n = inputs[0][0];
    auto i_d = inputs[0][1];
    auto i_h = inputs[0][2];
    auto i_w = inputs[0][3];
    auto i_c = inputs[0][4];

    bool is_ceil_mode = false;
    is_ceil_mode = params.count("ceil_mode") ? (bool)params["ceil_mode"] : false;

    auto infer_hw = [is_ceil_mode](Expr in, Expr k, Expr s, Expr d, Expr p0, Expr p1) -> Expr {
        auto new_k = k + (k - 1) * (d - 1);
        Expr out;
        if (is_ceil_mode)
        {
            // out = ((in - new_k + p0 + p1) / s).ceil() + 1;
            out = (in - new_k + p0 + p1).floordiv(s) + 1;
            auto p = Expr::select(((in - new_k + p0 + p1) % s) > 0, 1, 0);
            p = Expr::select((out * s + 1) <= (in + p0 + p1), p, 0);
            out = out + p;
        }
        else
        {
            out = (in - new_k + p0 + p1).floordiv(s) + 1;
        }
        return out;
    };
    auto od = infer_hw(i_d, kernel_z, stride_z, dilation_z, pad_z_begin, pad_z_end);
    auto oh = infer_hw(i_h, kernel_y, stride_y, dilation_y, pad_y_begin, pad_y_end);
    auto ow = infer_hw(i_w, kernel_x, stride_x, dilation_x, pad_x_begin, pad_x_end);

    SymbolShape s({i_n, od, oh, ow, i_c});
    std::vector<SymbolShape> ret = {s};
    return ret;
}

static std::vector<SymbolShape> infer_local_backward(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != 4 && inputs[0].dim() != 5)
    {
        return std::vector<SymbolShape>();
    }

    auto &params = n->params();
    if (!params.count("kernel_x") || !params.count("kernel_y") ||
        !params.count("stride_x") || !params.count("stride_y"))
    {
        return std::vector<SymbolShape>();
    }

    if (n->type() == OpType_Pooling &&
        params.count("ceil_mode") && (bool)params["ceil_mode"])
    {
        return std::vector<SymbolShape>();
    }

    int kernel_x = params["kernel_x"];
    int kernel_y = params["kernel_y"];
    int stride_x = params["stride_x"];
    int stride_y = params["stride_y"];
    int dilation_x = params["dilation_x"];
    int dilation_y = params["dilation_y"];
    int pad_left = params["pad_left"];
    int pad_right = params["pad_right"];
    int pad_top = params["pad_top"];
    int pad_bottom = params["pad_bottom"];

    auto i_n = inputs[0].n();
    auto i_h = inputs[0].h();
    auto i_w = inputs[0].w();
    Expr o_c = n->inputs()[0]->shape().c_total();

    auto infer_hw = [](Expr in, Expr k, Expr s, Expr d, Expr p_begin, Expr p_end) -> Expr {
        auto new_k = k + (k - 1) * (d - 1);
        return (in - 1) * s - p_begin - p_end + new_k;
    };
    auto oh = infer_hw(i_h, kernel_y, stride_y, dilation_y, pad_top, pad_bottom);
    auto ow = infer_hw(i_w, kernel_x, stride_x, dilation_x, pad_left, pad_right);

    std::vector<SymbolShape> ret = {SymbolShape({i_n, oh, ow, o_c})};
    handle_backward_layout(n, ret);

    return ret;
}

static std::vector<SymbolShape> infer_transpose(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<int> perm = n->params()["perm"];

    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        s.push_back(inputs[0][perm[i]]);
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_keep_shape(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (n->type() == OpType_Pow || n->type() == OpType_Logical || n->type() == OpType_BitShift)
    {
        if (inputs.size() != 2)
        {
            return std::vector<SymbolShape>();
        }

        if (inputs[0].dim() != inputs[1].dim())
        {
            return std::vector<SymbolShape>();
        }

        return {inputs[0]};
    }
    else if (n->type() == OpType_RotaryEmbedding)
    {
        return {inputs[0]};
    }

    return inputs;
}

static bool infer_broadcast_shape(std::vector<int> s0,
                                  std::vector<Expr> ss0,
                                  std::vector<int> s1,
                                  std::vector<Expr> ss1,
                                  std::vector<int> &s,
                                  std::vector<Expr> &ss)
{
    s.clear();
    ss.clear();
    auto max_size = std::max(s0.size(), s1.size());
    if (max_size == 0)
    {
        return true;
    }
    while (s0.size() < max_size)
    {
        s0.insert(s0.begin(), 1);
        ss0.insert(ss0.begin(), 1);
    }
    while (s1.size() < max_size)
    {
        s1.insert(s1.begin(), 1);
        ss1.insert(ss1.begin(), 1);
    }

    auto s0_size = TensorShape(s0).size();
    auto s1_size = TensorShape(s1).size();
    for (size_t i = 0; i < max_size; ++i)
    {
        if (s0[i] == 1 && s1[i] > 1)
        {
            s.push_back(s1[i]);
            ss.push_back(ss1[i]);
        }
        else if (s0[i] > 1 && s1[i] == 1)
        {
            s.push_back(s0[i]);
            ss.push_back(ss0[i]);
        }
        else if (s0[i] == s1[i])
        {
            auto is_ss0_constant = ss0[i]->is_constant();
            auto is_ss1_constant = ss1[i]->is_constant();
            s.push_back(s0[i]);
            if (is_ss0_constant && !is_ss1_constant)
            {
                ss.push_back(ss1[i]);
            }
            else if (!is_ss0_constant && is_ss1_constant)
            {
                ss.push_back(ss0[i]);
            }
            else if (s0_size < s1_size)
            {
                ss.push_back(ss1[i]);
            }
            else
            {
                ss.push_back(ss0[i]);
            }
        }
    }
    if (s.size() != max_size)
    {
        s.clear();
        ss.clear();
        return false;
    }
    return true;
}

static std::vector<SymbolShape> infer_broadcast_ops(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<int> s0 = n->inputs()[0]->shape();
    std::vector<int> s1 = n->inputs()[1]->shape();
    std::vector<Expr> ss0 = inputs[0];
    std::vector<Expr> ss1 = inputs[1];

    std::vector<int> s;
    std::vector<Expr> ss;
    if (!infer_broadcast_shape(s0, ss0, s1, ss1, s, ss))
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret = {SymbolShape(ss)};
    if (n->type() == OpType_DivMod)
    {
        ret.push_back(SymbolShape(ss));
    }
    return ret;
}

static std::vector<SymbolShape> infer_concat(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.empty())
    {
        return std::vector<SymbolShape>();
    }

    for (uint32_t i = 1; i < inputs.size(); ++i)
    {
        if (inputs[i].dim() != inputs[0].dim())
        {
            return std::vector<SymbolShape>();
        }
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    int axis = n->params()["axis"];
    if (axis < 0)
    {
        axis += inputs[0].dim();
    }

    Expr d = inputs[0][axis];
    for (uint32_t i = 1; i < inputs.size(); ++i)
    {
        d = d + inputs[i][axis];
    }

    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        if (i == axis)
        {
            s.push_back(d);
        }
        else
        {
            s.push_back(inputs[0][i]);
        }
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_constant(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    auto weights = n->constants()["weights"];

    std::vector<Expr> s;
    for (int i = 0; i < weights->shape().dim(); ++i)
    {
        s.push_back(weights->shape()[i]);
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_crop(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    std::vector<std::vector<Expr>> ds_crops;
    if (n->params().count("ds_crops"))
    {
        ds_crops = n->params()["ds_crops"];
    }
    else
    {
        std::vector<int> shape = n->inputs()[0]->shape();
        std::vector<std::vector<int>> crops = n->params()["crops"];
        // support [n,h,w,c]->[n,h,w,c/2]
        if (shape.size() == 4 &&
            shape[0] == crops[0][1] - crops[0][0] &&
            shape[1] == crops[1][1] - crops[1][0] &&
            shape[2] == crops[2][1] - crops[2][0] &&
            shape[3] / 2 == crops[3][1] - crops[3][0] &&
            inputs[0][3]->is_constant())
        {
            for (int i = 0; i < inputs[0].dim() - 1; ++i)
            {
                ds_crops.push_back({0, inputs[0][i]});
            }
            ds_crops.push_back({crops[3][0], crops[3][1]});
            n->params()["ds_crops"] = ds_crops;
        }
        // support [n,h,s,d,c]->[n,h,s,d,c/2]
        else if (shape.size() == 5 &&
                 shape[0] == crops[0][1] - crops[0][0] &&
                 shape[1] == crops[1][1] - crops[1][0] &&
                 shape[2] == crops[2][1] - crops[2][0] &&
                 shape[3] == crops[3][1] - crops[3][0] &&
                 shape[4] / 2 == crops[4][1] - crops[4][0] &&
                 inputs[0][4]->is_constant())
        {
            for (int i = 0; i < inputs[0].dim() - 1; ++i)
            {
                ds_crops.push_back({0, inputs[0][i]});
            }
            ds_crops.push_back({crops[4][0], crops[4][1]});
            n->params()["ds_crops"] = ds_crops;
        }
        // support [n,h,w,c]->[1,h,w,c]
        else if (shape.size() == 4 &&
                 shape[0] > 1 && crops[0][1] - crops[0][0] == 1 &&
                 shape[1] == crops[1][1] - crops[1][0] &&
                 shape[2] == crops[2][1] - crops[2][0] &&
                 shape[3] == crops[3][1] - crops[3][0] &&
                 inputs[0][0]->is_constant())
        {
            ds_crops.push_back({crops[0][0], crops[0][1]});
            for (int i = 1; i < inputs[0].dim(); ++i)
            {
                ds_crops.push_back({0, inputs[0][i]});
            }
            n->params()["ds_crops"] = ds_crops;
        }
        else
        {
            for (auto &crop : crops)
            {
                ds_crops.push_back({crop[0], crop[1]});
            }
            n->params()["ds_crops"] = ds_crops;
        }
    }

    if (ds_crops.size() != static_cast<uint32_t>(inputs[0].dim()))
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        s.push_back(ds_crops[i][1] - ds_crops[i][0]);
    }

    return std::vector<SymbolShape>({s});
}

static std::vector<SymbolShape> infer_fullconnected(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    Expr o_c = (int)n->params()["num_output"];

    auto o_n = inputs[0].n();

    return {SymbolShape({o_n, o_c})};
}

static std::vector<SymbolShape> infer_gather(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1 && inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    int axis = n->params()["axis"];
    int batch_dims = n->params()["batch_dims"];

    std::vector<Expr> lut_shape, idx_shape;
    if (inputs.size() == 1)
    {
        if (!n->constants().count("lut"))
        {
            return std::vector<SymbolShape>();
        }
        auto lut = n->constants()["lut"];

        if (n->params().count("method") && (std::string)n->params()["method"] == "idx_lut") // input0 is lut
        {
            for (int32_t i = 0; i < lut->shape().dim(); ++i)
            {
                idx_shape.push_back(lut->shape()[i]);
            }
            lut_shape = inputs[0];
        }
        else // input0 is idx
        {
            for (int32_t i = 0; i < lut->shape().dim(); ++i)
            {
                lut_shape.push_back(lut->shape()[i]);
            }
            idx_shape = inputs[0];
        }
    }
    else // input0 is lut, input1 is idx
    {
        lut_shape = inputs[0];
        idx_shape = inputs[1];
    }

    if (axis < 0)
    {
        axis += lut_shape.size();
    }
    if (batch_dims < 0)
    {
        batch_dims += idx_shape.size();
    }
    if (axis < batch_dims || axis >= (int)lut_shape.size() || batch_dims >= (int)idx_shape.size())
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s;
    for (int i = 0; i < (int)lut_shape.size(); ++i)
    {
        if (i == axis)
        {
            for (int j = batch_dims; j < (int)idx_shape.size(); ++j)
            {
                s.push_back(idx_shape[j]);
            }
        }
        else
        {
            s.push_back(lut_shape[i]);
        }
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_global_pool(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    auto &params = n->params();
    bool keep_dims = false;
    keep_dims = params.count("keep_dims") ? (bool)params["keep_dims"] : false;

    auto o_n = inputs[0].n();
    auto o_c = inputs[0].c();
    if (keep_dims)
    {
        return {SymbolShape({o_n, 1, 1, o_c})};
    }
    else
    {
        return {SymbolShape({o_n, o_c})};
    }
}

static std::vector<SymbolShape> infer_global_pool_backward(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    auto &params = n->params();
    bool keep_dims = false;
    keep_dims = params.count("keep_dims") ? (bool)params["keep_dims"] : false;

    auto o_n = inputs[0].n();
    Expr o_c;
    if (!keep_dims)
    {
        o_c = inputs[0].h();
    }
    else
    {
        o_c = inputs[0].c_total();
    }

    // always return the minimum input shape
    std::vector<SymbolShape> ret = {SymbolShape({o_n, 1, 1, o_c})};
    handle_backward_layout(n, ret);

    return ret;
}

static std::vector<SymbolShape> infer_matmul(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != inputs[1].dim())
    {
        return std::vector<SymbolShape>();
    }

    bool trans_a = n->params()["trans_a"];
    bool trans_b = n->params()["trans_b"];
    auto dims = inputs[0].dim();

    std::vector<Expr> s;
    for (int i = 0; i < dims - 2; ++i)
    {
        s.push_back(inputs[0][i].max(inputs[1][i]));
    }
    if (trans_a)
    {
        s.push_back(inputs[0][dims - 1]);
    }
    else
    {
        s.push_back(inputs[0][dims - 2]);
    }
    if (trans_b)
    {
        s.push_back(inputs[1][dims - 2]);
    }
    else
    {
        s.push_back(inputs[1][dims - 1]);
    }

    if (n->attrs().count(Constants::c_align))
    {
        int alignment = n->get_attr(Constants::c_align);
        s[dims - 1] = (s[dims - 1] + alignment - 1).floordiv(alignment) * alignment;
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_pad(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<std::vector<Expr>> ds_pads;
    if (n->params().count("ds_pads"))
    {
        ds_pads = n->params()["ds_pads"];
    }
    else
    {
        std::vector<std::vector<int>> pads = n->params()["pads"];
        for (auto &pad : pads)
        {
            ds_pads.push_back({pad[0], pad[1]});
        }
    }

    if (ds_pads.size() != static_cast<uint32_t>(inputs[0].dim()))
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        s.push_back(inputs[0][i] + ds_pads[i][0] + ds_pads[i][1]);
    }
    return {SymbolShape(s)};
}

static void get_subsets(const std::vector<int> &arr,
                        size_t begin,
                        std::vector<int> &tmp,
                        std::vector<std::vector<int>> &res,
                        size_t k)
{
    if (tmp.size() >= k)
    {
        res.push_back(tmp);
    }
    for (size_t i = begin; i < arr.size(); ++i)
    {
        tmp.push_back(arr[i]);
        get_subsets(arr, i + 1, tmp, res, k);
        tmp.pop_back();
    }
}

static std::vector<std::vector<int>> get_combs(const std::vector<int> &arr)
{
    std::vector<std::vector<int>> ret;
    std::vector<int> tmp;
    get_subsets(arr, 0, tmp, ret, 2);
    return ret;
}

static bool auto_infer_reshape(NodePtr n,
                               const std::vector<SymbolShape> &inputs,
                               std::vector<Expr> &s,
                               bool backward = false)
{
    std::vector<int> in_shape;
    std::vector<int> out_shape;
    if (backward)
    {
        in_shape = n->params()["shape"];
        out_shape = n->inputs()[0]->shape();
    }
    else
    {
        in_shape = n->inputs()[0]->shape();
        out_shape = n->params()["shape"];
    }

    // [] -> [1]
    // [1] -> []
    if (in_shape.empty() || out_shape.empty())
    {
        s.clear();
        for (auto d : out_shape)
        {
            s.push_back(d);
        }
        return true;
    }

    std::map<int, bool> ran_in_dims;
    std::map<int, bool> ran_out_dims;
    for (size_t i = 0; i < in_shape.size(); ++i)
    {
        ran_in_dims[i] = in_shape[i] == 1 ? true : false;
    }
    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        ran_out_dims[i] = out_shape[i] == 1 ? true : false;
    }

    std::map<int, std::vector<int>> merge_remap;
    std::map<int, int> trans_remap;

    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        if (ran_out_dims[i])
        {
            continue;
        }

        // find merged input dimensions
        std::vector<int> in_dims;
        for (size_t j = 0; j < in_shape.size(); ++j)
        {
            if (ran_in_dims[j] || in_shape[j] >= out_shape[i])
            {
                continue;
            }
            in_dims.push_back(j);
        }
        auto in_combs = get_combs(in_dims);
        std::vector<std::vector<int>> valid_combs;
        for (auto &comb : in_combs)
        {
            int val = 1;
            for (auto d : comb)
            {
                val *= in_shape[d];
            }
            if (val == out_shape[i])
            {
                valid_combs.push_back(comb);
            }
        }
        if (valid_combs.size())
        {
            merge_remap[i] = valid_combs[0];
            for (auto d : valid_combs[0])
            {
                ran_in_dims[d] = true;
            }
            ran_out_dims[i] = true;
            continue;
        }

        // find transposed input dimensions
        std::vector<int> trans_dims;
        for (size_t j = 0; j < in_shape.size(); ++j)
        {
            if (ran_in_dims[j] || in_shape[j] != out_shape[i])
            {
                continue;
            }
            trans_dims.push_back(j);
        }
        if (trans_dims.size())
        {
            trans_remap[i] = trans_dims[0];
            ran_in_dims[trans_dims[0]] = true;
            ran_out_dims[i] = true;
            continue;
        }
    }

    // check reshape conflict
    for (size_t i = 0; i < in_shape.size(); ++i)
    {
        if (!ran_in_dims[i] && !inputs[0][i]->is_constant())
        {
            return false;
        }
    }

    // assemble input dimensions
    s.resize(out_shape.size());
    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        s[i] = out_shape[i];
    }
    for (auto &p : merge_remap)
    {
        Expr prod = 1;
        for (auto d : p.second)
        {
            prod = prod * inputs[0][d];
        }
        s[p.first] = prod;
    }
    for (auto &p : trans_remap)
    {
        s[p.first] = inputs[0][p.second];
    }

    return true;
}

static std::vector<SymbolShape> infer_reshape(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    std::vector<Expr> s;

    if (n->params().count("ds_shape"))
    {
        s = n->params()["ds_shape"];
        return {SymbolShape(s)};
    }

    if (auto_infer_reshape(n, inputs, s))
    {
        return {SymbolShape(s)};
    }

    return std::vector<SymbolShape>();
}

static std::vector<SymbolShape> infer_reshape_backward(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    // std::vector<Expr> s;
    // if (auto_infer_reshape(n, inputs, s, true))
    // {
    //     return {SymbolShape(s)};
    // }

    return std::vector<SymbolShape>();
}

static std::vector<SymbolShape> infer_slice(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() == 5 && !n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() != 1 && inputs.size() != 5)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;

    if (inputs.size() == 1)
    {
        if (n->params().count("ds_begin"))
        {
            std::vector<Expr> begin = n->params()["ds_begin"];
            std::vector<Expr> end = n->params()["ds_end"];
            std::vector<Expr> strides = n->params()["ds_strides"];

            std::vector<Expr> s(begin.size());
            for (uint32_t i = 0; i < begin.size(); ++i)
            {
                s[i] = (end[i] - begin[i] - 1) / strides[i] + 1;
            }
            ret = {SymbolShape(s)};
        }
        else
        {
            std::vector<int> begin = n->params()["begin"];
            std::vector<int> end = n->params()["end"];
            std::vector<int> strides = n->params()["strides"];

            std::vector<Expr> s(begin.size());
            for (uint32_t i = 0; i < begin.size(); ++i)
            {
                s[i] = (end[i] - begin[i] - 1) / strides[i] + 1;
            }
            ret = {SymbolShape(s)};
        }
    }
    else
    {
        if (n->attrs().count(Constants::output_symbol_shape))
        {
            ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
        }
        else
        {
            std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
            int ndim = n->inputs()[0]->shape().dim();
            ret.push_back(SymbolShape(da_symbols));
            ret.push_back(SymbolShape({ndim}));
            n->set_attr(Constants::output_symbol_shape, ret);
        }
    }

    return ret;
}

static std::vector<SymbolShape> infer_split(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    int axis = n->params()["axis"];
    if (axis < 0)
    {
        axis += inputs[0].dim();
    }
    if (axis >= inputs[0].dim())
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> ds_splits;
    if (n->params().count("ds_splits"))
    {
        ds_splits = n->params()["ds_splits"];
    }
    else
    {
        std::vector<int> splits = n->params()["splits"];
        for (auto split : splits)
        {
            ds_splits.push_back(split);
        }
    }

    std::vector<SymbolShape> ret;
    for (auto split : ds_splits)
    {
        std::vector<Expr> s;
        for (int i = 0; i < inputs[0].dim(); ++i)
        {
            if (i == axis)
            {
                s.push_back(split);
            }
            else
            {
                s.push_back(inputs[0][i]);
            }
        }
        ret.push_back(SymbolShape(s));
    }

    return ret;
}

static std::vector<SymbolShape> infer_tile(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> ds_repeats;
    if (n->params().count("ds_repeats"))
    {
        ds_repeats = n->params()["ds_repeats"];
    }
    else
    {
        std::vector<int> repeats = n->params()["repeats"];
        for (auto r : repeats)
        {
            ds_repeats.push_back(r);
        }
    }

    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        s.push_back(inputs[0][i] * ds_repeats[i]);
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_nonzero(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (!n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;

    if (n->attrs().count(Constants::output_symbol_shape))
    {
        ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
    }
    else
    {
        std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
        Expr b = da_symbols[0];
        Expr d = inputs[0].dim();
        ret.push_back(SymbolShape({d, b}));
        ret.push_back(SymbolShape({1}));
        n->set_attr(Constants::output_symbol_shape, ret);
    }

    return ret;
}

static std::vector<SymbolShape> infer_scatternd(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 3)
    {
        return std::vector<SymbolShape>();
    }

    return {inputs[0]};
}

static std::vector<SymbolShape> infer_upsample(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != 4)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s;

    if (n->type() == OpType_BatchToSpace ||
        n->type() == OpType_DepthToSpace)
    {
        int block_size_x = n->params()["block_size_x"];
        int block_size_y = n->params()["block_size_y"];

        if (n->type() == OpType_BatchToSpace)
        {
            int crop_left = n->params()["crop_left"];
            int crop_right = n->params()["crop_right"];
            int crop_top = n->params()["crop_top"];
            int crop_bottom = n->params()["crop_bottom"];

            s = {
                inputs[0].n().floordiv(block_size_x * block_size_y),
                inputs[0].h() * block_size_y - crop_top - crop_bottom,
                inputs[0].w() * block_size_x - crop_left - crop_right,
                inputs[0].c(),
            };
        }
        else
        {
            s = {
                inputs[0].n(),
                inputs[0].h() * block_size_y,
                inputs[0].w() * block_size_x,
                inputs[0].c().floordiv(block_size_x * block_size_y),
            };
        }
    }
    else if (n->type() == OpType_Resize)
    {
        double ratio_x = (double)n->outputs()[0]->shape().w() / (double)n->inputs()[0]->shape().w();
        double ratio_y = (double)n->outputs()[0]->shape().h() / (double)n->inputs()[0]->shape().h();
        if (n->params().count("ratio_x"))
        {
            ratio_x = (double)n->params()["ratio_x"];
        }
        if (n->params().count("ratio_y"))
        {
            ratio_y = (double)n->params()["ratio_y"];
        }

        s = {
            inputs[0].n(),
            (inputs[0].h() * ratio_y).floor(),
            (inputs[0].w() * ratio_x).floor(),
            inputs[0].c(),
        };
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_reduce(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s = inputs[0];

    std::vector<int> axis = n->params()["axis"];
    for (auto d : axis)
    {
        if (d < 0)
        {
            d += inputs[0].dim();
        }
        s[d] = 1;
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_query_rebatch(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() < 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> args;
    for (size_t i = 2; i < inputs.size(); ++i)
    {
        args.push_back(inputs[i][0]);
    }
    auto max_len = inputs[1][0].max(args);
    std::vector<Expr> s = {
        inputs[0][0],
        inputs.size() - 1,
        max_len,
        inputs[0][2],
    };

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_slot_update(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() < 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s = {
        inputs[0][0],
        n->outputs()[0]->shape()[1],
        inputs[0][3],
    };

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_grid_sample(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != 4 || inputs[1].dim() != 4)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s = {
        inputs[0][0],
        inputs[1][1],
        inputs[1][2],
        inputs[0][3],
    };

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_layoutcvt(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::string from = n->params()["from"];
    std::string to = n->params()["to"];
    if (datalayout_to_string(inputs[0].layout()) != from)
    {
        return std::vector<SymbolShape>();
    }

    auto in_shape = n->inputs()[0]->shape();
    in_shape.symbol() = SymbolShape(inputs[0], inputs[0].layout(), inputs[0].indexing());
    if (!in_shape.change_layout(string_to_datalayout(to)) && from != to)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s = in_shape.symbol();

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_cache_update(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (!n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() != 3)
    {
        return std::vector<SymbolShape>();
    }

    if ((inputs[0].dim() != 3 && inputs[0].dim() != 4) || inputs[2].dim() != 4)
    {
        return std::vector<SymbolShape>();
    }

    int axis = 2;
    if (n->params().count("axis"))
    {
        axis = n->params()["axis"];
    }
    if (axis != 2 && axis != 3)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;

    if (n->attrs().count(Constants::output_symbol_shape))
    {
        ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
    }
    else
    {
        std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
        Expr tok_len = da_symbols[0] + inputs[2][2];
        std::vector<Expr> s;
        if (axis == 3)
        {
            s = {
                inputs[2][0],
                inputs[2][1],
                inputs[2][2],
                tok_len,
            };
        }
        else
        {
            s = {
                inputs[2][0],
                inputs[2][1],
                tok_len,
                inputs[2][3],
            };
        }
        ret = {SymbolShape(s), inputs[0]};
        n->set_attr(Constants::output_symbol_shape, ret);
    }

    return ret;
}

static std::vector<SymbolShape> infer_s2b(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != 4)
    {
        return std::vector<SymbolShape>();
    }

    int block_size_x = n->params()["block_size_x"];
    int block_size_y = n->params()["block_size_y"];
    int pad_top = n->params()["pad_top"];
    int pad_bottom = n->params()["pad_bottom"];
    int pad_left = n->params()["pad_left"];
    int pad_right = n->params()["pad_right"];

    auto in = inputs[0].n();
    auto ih = inputs[0].h();
    auto iw = inputs[0].w();

    std::vector<Expr> s = {
        in * block_size_x * block_size_y,
        (ih + pad_top + pad_bottom).floordiv(block_size_y),
        (iw + pad_left + pad_right).floordiv(block_size_x),
        inputs[0].c(),
    };

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_s2d(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }
    if (inputs[0].dim() != 4)
    {
        return std::vector<SymbolShape>();
    }

    int block_size_y = n->params()["block_size_y"];
    int block_size_x = n->params()["block_size_x"];
    std::vector<Expr> s = {
        inputs[0].n(),
        inputs[0].h().floordiv(block_size_y),
        inputs[0].w().floordiv(block_size_x),
        inputs[0].c() * block_size_y * block_size_x,
    };
    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_layernorm(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret = {inputs[0]};
    if (n->outputs().size() == 3)
    {
        std::vector<Expr> s;
        std::vector<int> axis = n->params()["axis"];
        for (int i = 0; i < inputs[0].dim(); ++i)
        {
            if (contains(axis, i))
            {
                s.push_back(1);
            }
            else
            {
                s.push_back(inputs[0][i]);
            }
        }
        ret.push_back(SymbolShape(s));
        ret.push_back(SymbolShape(s));
    }

    return ret;
}

static std::vector<SymbolShape> infer_moments(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> s;
    std::vector<int> axis = n->params()["axis"];
    for (size_t i = 0; i < axis.size(); ++i)
    {
        if (axis[i] < 0)
        {
            axis[i] += inputs[0].dim();
        }
    }
    bool keepdims = n->params()["keepdims"];

    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        if (contains(axis, i))
        {
            if (keepdims)
            {
                s.push_back(1);
            }
        }
        else
        {
            s.push_back(inputs[0][i]);
        }
    }

    return {SymbolShape(s), SymbolShape(s)};
}

static std::vector<SymbolShape> infer_gathernd(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    int batch_dims = n->params()["batch_dims"];
    int lut_start_dim = batch_dims + n->inputs()[1]->shape()[-1];

    std::vector<Expr> lut_shape = inputs[0];
    std::vector<Expr> idx_shape = inputs[1];

    std::vector<Expr> s;
    for (int i = 0; i < inputs[1].dim() - 1; ++i)
    {
        s.push_back(idx_shape[i]);
    }
    for (int i = lut_start_dim; i < inputs[0].dim(); ++i)
    {
        s.push_back(lut_shape[i]);
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_roialign(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != 4 || inputs[1].dim() != 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<int> pooled_shape = n->params()["pooled_shape"];

    std::vector<Expr> s = {
        inputs[1][0],
        pooled_shape[0],
        pooled_shape[1],
        inputs[0][3],
    };

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_output_shape(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    int ndim = inputs[0].dim();
    int start = n->params()["start"];
    int end = n->params()["end"];
    if (start < 0)
    {
        start += ndim;
    }
    if (end < 0)
    {
        end += ndim;
    }
    std::vector<Expr> s = {end - start};

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_const_shape(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (!n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;

    if (n->attrs().count(Constants::output_symbol_shape))
    {
        ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
    }
    else
    {
        std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
        ret = {SymbolShape(da_symbols)};
        n->set_attr(Constants::output_symbol_shape, ret);
    }

    return ret;
}

static std::vector<SymbolShape> infer_forrange_head(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    size_t depend_input_num = (int)get_control_param(n, "dependency_inputs_num");
    size_t scan_output_num = (int)get_control_param(n, "scan_outputs_num");

    if (inputs.size() < 2 + depend_input_num)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret = {
        SymbolShape({1}),
        SymbolShape({1}),
        SymbolShape({scan_output_num}),
        inputs[0],
        inputs[0],
        inputs[1],
    };

    for (size_t i = 2; i < inputs.size(); ++i)
    {
        ret.push_back(inputs[i]);
    }

    return ret;
}

static std::vector<SymbolShape> infer_forrange_end(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (!n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    size_t depend_input_num = (int)get_control_param(n, "dependency_inputs_num");
    size_t scan_output_num = (int)get_control_param(n, "scan_outputs_num");

    if (inputs.size() != 7 + 2 * depend_input_num + scan_output_num)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;

    if (n->attrs().count(Constants::output_symbol_shape))
    {
        ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
    }
    else
    {
        std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
        for (size_t i = 0; i < depend_input_num; ++i)
        {
            ret.push_back(inputs[i + 7 + depend_input_num]);
        }
        for (size_t i = depend_input_num; i < n->outputs().size(); ++i)
        {
            auto in_shape = inputs[i + 7 + depend_input_num];
            std::vector<Expr> s = {da_symbols[i - depend_input_num]};
            int start_dim = in_shape.dim() == n->outputs()[i]->shape().dim() ? 1 : 0;
            for (int j = start_dim; j < in_shape.dim(); ++j)
            {
                s.push_back(in_shape[j]);
            }
            ret.push_back(SymbolShape(s));
        }
        n->set_attr(Constants::output_symbol_shape, ret);
    }

    return ret;
}

static std::vector<SymbolShape> infer_if_head(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    size_t then_input_num = (int)get_control_param(n, "then_branch_inputs_num");
    size_t else_input_num = (int)get_control_param(n, "else_branch_inputs_num");

    if (inputs.size() != 1 + then_input_num + else_input_num)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret = {
        SymbolShape({1}),
        inputs[0],
    };

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        ret.push_back(inputs[i]);
    }

    return ret;
}

static std::vector<SymbolShape> infer_if_end(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (!n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    size_t output_num = n->outputs().size();

    if (inputs.size() != 2 + 2 * output_num)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<Expr> cst_exprs;
    for (size_t i = 2; i < 2 + output_num; ++i)
    {
        auto s0 = inputs[i];
        auto s1 = inputs[i + output_num];
        for (int j = 0; j < s0.dim(); ++j)
        {
            if (s0[j]->is_constant() && s0[j]->equal(s1[j]))
            {
                cst_exprs.push_back(s0[j]);
            }
            else
            {
                cst_exprs.push_back(nullptr);
            }
        }
    }

    std::vector<SymbolShape> ret;

    if (n->attrs().count(Constants::output_symbol_shape))
    {
        ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
    }
    else
    {
        std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
        uint32_t index = 0;
        for (size_t i = 0; i < output_num; ++i)
        {
            std::vector<Expr> s;
            for (int j = 0; j < n->outputs()[i]->shape().dim(); ++j)
            {
                if (cst_exprs[index])
                {
                    s.push_back(cst_exprs[index]);
                }
                else
                {
                    s.push_back(da_symbols[index]);
                }
                index++;
            }
            ret.push_back(SymbolShape(s));
        }
        n->set_attr(Constants::output_symbol_shape, ret);
    }

    return ret;
}

static std::vector<SymbolShape> infer_topk(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() == 2 && !n->params().count("ds_data_aware_symbol"))
    {
        return std::vector<SymbolShape>();
    }

    if (inputs.size() != 1 && inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    int axis = n->params()["axis"];
    if (axis < 0)
    {
        axis += inputs[0].dim();
    }

    std::vector<SymbolShape> ret;

    if (inputs.size() == 1)
    {
        Expr k = (int)n->params()["k"];
        std::vector<Expr> s = inputs[0];
        s[axis] = k;
        ret = {SymbolShape(s), SymbolShape(s)};
    }
    else
    {
        if (n->attrs().count(Constants::output_symbol_shape))
        {
            ret = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
        }
        else
        {
            std::vector<Expr> da_symbols = n->params()["ds_data_aware_symbol"];
            std::vector<Expr> s = inputs[0];
            s[axis] = da_symbols[0];
            ret = {SymbolShape(s), SymbolShape(s)};
            n->set_attr(Constants::output_symbol_shape, ret);
        }
    }

    return ret;
}

static std::vector<SymbolShape> infer_scatter(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 3)
    {
        return std::vector<SymbolShape>();
    }

    if (inputs[0].dim() != inputs[1].dim() && inputs[0].dim() != inputs[2].dim())
    {
        return std::vector<SymbolShape>();
    }

    return {inputs[0]};
}

static std::vector<SymbolShape> infer_gather_elements(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    return {inputs[1]};
}

static std::vector<SymbolShape> infer_where(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 3)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<int> s0 = n->inputs()[0]->shape();
    std::vector<int> s1 = n->inputs()[1]->shape();
    std::vector<int> s2 = n->inputs()[2]->shape();
    std::vector<Expr> ss0 = inputs[0];
    std::vector<Expr> ss1 = inputs[1];
    std::vector<Expr> ss2 = inputs[2];

    std::vector<int> s_tmp;
    std::vector<Expr> ss_tmp;
    if (!infer_broadcast_shape(s0, ss0, s1, ss1, s_tmp, ss_tmp))
    {
        return std::vector<SymbolShape>();
    }

    std::vector<int> s;
    std::vector<Expr> ss;
    if (!infer_broadcast_shape(s_tmp, ss_tmp, s2, ss2, s, ss))
    {
        return std::vector<SymbolShape>();
    }

    return {SymbolShape(ss)};
}

static std::vector<SymbolShape> infer_channel_shuffle(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 1)
    {
        return std::vector<SymbolShape>();
    }
    int axis = inputs[0].dim() - 1;
    int32_t splits = n->params()["splits"];
    std::vector<SymbolShape> ret;
    std::vector<Expr> s;
    for (int i = 0; i < inputs[0].dim(); ++i)
    {
        if (i == axis)
        {
            s.push_back(inputs[0][i].floordiv(splits));
        }
        else
        {
            s.push_back(inputs[0][i]);
        }
    }
    for (int32_t i = 0; i < splits; ++i)
    {
        ret.push_back(SymbolShape(s));
    }

    return ret;
}

static std::vector<SymbolShape> infer_switch(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() < 1)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    std::vector<Expr> s0 = inputs[0];
    std::vector<std::vector<Expr>> ss(s0.size());
    for (size_t i = 1; i < inputs.size(); ++i)
    {
        std::vector<Expr> s = inputs[i];
        if (s.size() != s0.size())
        {
            return std::vector<SymbolShape>();
        }

        for (size_t j = 0; j < s.size(); ++j)
        {
            ss[j].push_back(s[j]);
        }
    }

    std::vector<Expr> s;
    for (size_t i = 0; i < ss.size(); ++i)
    {
        s.push_back(s0[i].max(ss[i]));
    }

    return {SymbolShape(s)};
}

static std::vector<SymbolShape> infer_empty(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() > 0)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    return std::vector<SymbolShape>();
}

static std::vector<SymbolShape> infer_repeat(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    if (inputs.size() != 2)
    {
        return std::vector<SymbolShape>();
    }

    std::vector<SymbolShape> ret;
    if (auto_infer_shape(n, inputs, ret))
    {
        return ret;
    }

    return std::vector<SymbolShape>();
}

static std::vector<SymbolShape> infer_sort(NodePtr n, const std::vector<SymbolShape> &inputs)
{
    std::vector<SymbolShape> ret;
    for (size_t i = 0; i < n->outputs().size(); ++i)
    {
        ret.push_back(inputs[0]);
    }
    return ret;
}

INFERSHAPE_REGISTER(OpType_Abs, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Acos, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Acosh, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Activation, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Add, infer_broadcast_ops);
INFERSHAPE_REGISTER(OpType_Asin, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Asinh, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_BNLL, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_BatchNorm, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_BiasAdd, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Cast, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Ceil, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Clip, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Concat, infer_concat);
INFERSHAPE_REGISTER(OpType_Constant, infer_constant);
INFERSHAPE_REGISTER(OpType_InternalConstant, infer_constant);
INFERSHAPE_REGISTER(OpType_Convolution, infer_local, infer_local_backward);
INFERSHAPE_REGISTER(OpType_Cosh, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Cosine, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Crop, infer_crop);
INFERSHAPE_REGISTER(OpType_InternalCrop, infer_crop);
INFERSHAPE_REGISTER(OpType_Cumulate, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_DepthwiseConv, infer_local);
INFERSHAPE_REGISTER(OpType_DeQuantize, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Dilation, infer_local);
INFERSHAPE_REGISTER(OpType_Eltwise, infer_eltwise, infer_eltwise_backward);
INFERSHAPE_REGISTER(OpType_Erf, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Erosion, infer_local);
INFERSHAPE_REGISTER(OpType_Exp, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Floor, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_FullyConnected, infer_fullconnected);
INFERSHAPE_REGISTER(OpType_Gather, infer_gather);
INFERSHAPE_REGISTER(OpType_GlobalPool, infer_global_pool, infer_global_pool_backward);
INFERSHAPE_REGISTER(OpType_GroupNorm, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_InstanceNorm, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Normalization, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_LRN, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_LayerNorm, infer_layernorm);
INFERSHAPE_REGISTER(OpType_BitShift, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Log, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_LogSoftmax, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Logical, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_MatMul, infer_matmul);
INFERSHAPE_REGISTER(OpType_MaxPoolingWithArgMax, infer_local);
INFERSHAPE_REGISTER(OpType_MVN, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Mul, infer_broadcast_ops);
INFERSHAPE_REGISTER(OpType_Negative, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Pad, infer_pad);
INFERSHAPE_REGISTER(OpType_InternalPad, infer_pad);
INFERSHAPE_REGISTER(OpType_Pooling, infer_local, infer_local_backward);
INFERSHAPE_REGISTER(OpType_Pooling3D, infer_pooling3d);
INFERSHAPE_REGISTER(OpType_Quantize, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Reciprocal, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Reshape, infer_reshape, infer_reshape_backward);
INFERSHAPE_REGISTER(OpType_RMSNorm, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Round, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Rsqrt, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Sine, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Sinh, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Slice, infer_slice);
INFERSHAPE_REGISTER(OpType_Softmax, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Split, infer_split);
INFERSHAPE_REGISTER(OpType_InternalSpliter, infer_split);
INFERSHAPE_REGISTER(OpType_Sqrt, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Square, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Sub, infer_broadcast_ops);
INFERSHAPE_REGISTER(OpType_Tan, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Tile, infer_tile);
INFERSHAPE_REGISTER(OpType_Transpose, infer_transpose);
INFERSHAPE_REGISTER(OpType_Trunc, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_NonZero, infer_nonzero);
INFERSHAPE_REGISTER(OpType_ScatterND, infer_scatternd);
INFERSHAPE_REGISTER(OpType_BatchToSpace, infer_upsample);
INFERSHAPE_REGISTER(OpType_DepthToSpace, infer_upsample);
INFERSHAPE_REGISTER(OpType_Resize, infer_upsample);
INFERSHAPE_REGISTER(OpType_Reduce, infer_reduce);
INFERSHAPE_REGISTER(OpType_QueryRebatch, infer_query_rebatch);
INFERSHAPE_REGISTER(OpType_SlotUpdate, infer_slot_update);
INFERSHAPE_REGISTER(OpType_GridSample, infer_grid_sample);
INFERSHAPE_REGISTER(OpType_LayoutConvertor, infer_layoutcvt);
INFERSHAPE_REGISTER(OpType_CacheUpdate, infer_cache_update);
INFERSHAPE_REGISTER(OpType_Div, infer_broadcast_ops);
INFERSHAPE_REGISTER(OpType_ConvTranspose, infer_local);
INFERSHAPE_REGISTER(OpType_SpaceToBatch, infer_s2b);
INFERSHAPE_REGISTER(OpType_SpaceToDepth, infer_s2d);
INFERSHAPE_REGISTER(OpType_SquaredDifference, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Pow, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Moments, infer_moments);
INFERSHAPE_REGISTER(OpType_GatherND, infer_gathernd);
INFERSHAPE_REGISTER(OpType_RoiAlign, infer_roialign);
INFERSHAPE_REGISTER(OpType_Dummy, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Shape, infer_output_shape);
INFERSHAPE_REGISTER(OpType_ConstantOfShape, infer_const_shape);
INFERSHAPE_REGISTER(OpType_ForRangeHeader, infer_forrange_head);
INFERSHAPE_REGISTER(OpType_ForRangeEnder, infer_forrange_end);
INFERSHAPE_REGISTER(OpType_IfHeader, infer_if_head);
INFERSHAPE_REGISTER(OpType_IfEnder, infer_if_end);
INFERSHAPE_REGISTER(OpType_TopK, infer_topk);
INFERSHAPE_REGISTER(OpType_ScatterElements, infer_scatter);
INFERSHAPE_REGISTER(OpType_DivMod, infer_broadcast_ops);
INFERSHAPE_REGISTER(OpType_GatherElements, infer_gather_elements);
INFERSHAPE_REGISTER(OpType_Where, infer_where);
INFERSHAPE_REGISTER(OpType_ChannelShuffle, infer_channel_shuffle);
INFERSHAPE_REGISTER(OpType_Switch, infer_switch);
INFERSHAPE_REGISTER(OpType_InternalSwitch, infer_switch);
INFERSHAPE_REGISTER(OpType_Empty, infer_empty);
INFERSHAPE_REGISTER(OpType_Repeat, infer_repeat);
INFERSHAPE_REGISTER(OpType_RotaryEmbedding, infer_keep_shape);
INFERSHAPE_REGISTER(OpType_Sort, infer_sort);
} // namespace tongue

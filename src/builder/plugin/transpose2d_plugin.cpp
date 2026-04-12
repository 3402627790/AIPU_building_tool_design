
#include "transpose2d_plugin.h"

using namespace aipubt;


const GraphMatcherPattern Transpose2DPlugin::get_graph_pattern()
{
    return {{{"transpose", OpType_Transpose}}, {}};
}
const std::string Transpose2DPlugin::get_layer_type_name()
{
    return VAR_NAME_TO_STR(OpType_Transpose);
}
uint32_t Transpose2DPlugin::get_score()
{
    return 5;
}
bool Transpose2DPlugin::set_target(const char *target)
{
    m_target_name = target;
    to_lower(m_target_name);
    if(!check_target("transpose 2d", m_target_name))
    {
        return false;
    }
    if((m_target_name.find("z2") ==  m_target_name.npos)
      & (m_target_name.find("z3") ==  m_target_name.npos)
      & (m_target_name.find("x1") ==  m_target_name.npos))
    {
        return false;
    }
    return true;
}
void Transpose2DPlugin::setup(BaseNodePtr node, BaseNodePtrList match_list)
{
    /* deny_layouts_comb */
    node->set_attr("keeping_layout", true); // do not support layout convert.
    LayoutCombinations valid_in = {
        {DataLayout_Flat},
        {DataLayout_NHWC},
        {DataLayout_NCHWC32},
        {DataLayout_NCHWC16},
        {DataLayout_NDCHWC32},
        {DataLayout_NDCHWC16},
        {DataLayout_NDHWC},
    };
    LayoutCombinations valid_out = {
        {DataLayout_Flat},
        {DataLayout_NHWC},
        {DataLayout_NCHWC32},
        {DataLayout_NCHWC16},
        {DataLayout_NDCHWC32},
        {DataLayout_NDCHWC16},
        {DataLayout_NDHWC},
    };
    node->set_attr("valid_input_layouts", valid_in);
    node->set_attr("valid_output_layouts", valid_out);

}
BuilderParams Transpose2DPlugin::generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
{
    DEBUG() << "Transpose2D is in use.";
    auto node = nodes[0];
    BuilderParams ro;
    std::vector<int> perm = nodes[0]->params()["perm"];
    TensorShape sh = sgnode->inputs()[0]->shape();
    _modify_perm(sh, perm);
    std::vector<int> shape;
    for(int i = 0; i < sh.dim(); i++)
    {
        shape.push_back(sh[i]);
    }
    _optimize_perm(shape, perm);
    DEBUG() << "shape" << NodeParamValue(shape) << " perm:" <<  NodeParamValue(perm);
    ro.push_back(sgnode->inputs()[0]);
    ro.push_back(sgnode->outputs()[0]);
    ro.push_back(shape[1]);
    ro.push_back(shape[0]);
    int sram_size = std::max(TARGET_TO_GSRAM_SIZE[m_target_name], 0x400);
    ro.push_back(sram_size);
    return ro;
}
const std::string Transpose2DPlugin::generate_code_name(BaseNodePtr, BaseNodePtrList &node)
{
    std::string name  = gen_bin_name("tpcc_transpose2d_", Tensor::INT8, m_target_name, false);
    return name;
}
const std::string Transpose2DPlugin::generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
{
    return "";
}

bool Transpose2DPlugin::_special_check_params(BaseNodePtrList nodes)
{
    std::vector<int> perm = nodes[0]->params()["perm"];
    TensorShape sh = nodes[0]->inputs()[0]->shape();
    std::vector<int> shape;
    for(int i = 0; i < sh.dim(); i++)
    {
        shape.push_back(sh[i]);
    }
    _optimize_perm(shape, perm);
    if(perm.size() != 2)
    {
        return false;
    }
    for(uint i = 0; i < shape.size(); ++i)
    {
        if(shape[i] > DMA_MAX_STRIDE_V2)
        {
            return false;
        }
    }
    if(nodes[0]->inputs()[0]->type() > 3)
    {
        return false;
    }
    return true;
}

void Transpose2DPlugin::_modify_perm(const TensorShape& sh, std::vector<int>& perm)
{
    DataLayout lo = sh.layout();
    if(lo == DataLayout_NCHWC32 ||
       lo == DataLayout_NCHWC16 ||
       lo == DataLayout_NCDHWC32 ||
       lo == DataLayout_NCDHWC16)
    {
        /* change 3 in perm to 1, 4 is fine, and then 1 = 2, 2 = 3.
         * i.e. [0,3,1,2]@NHW = [0,1,4,2,3]@NCHW32.
         * TensorShape.dim() is based on real data arrangement.
         */
        if(perm.size() + 1 == (uint)sh.dim())
        {
            for(uint i = 0; i < perm.size(); ++i)
            {

                if(perm[i] != 0)
                {
                    ++perm[i];
                }
            }
            for(uint i = 0; i < perm.size(); ++i)
            {
                if(perm[i] == (int)perm.size() - 1)
                {
                    /* after insert, the i is pointing to the inserted position. */
                    perm.insert(perm.begin() + i++, 1);
                    break;
                }
            }
        }
    }
}

void Transpose2DPlugin::_optimize_perm(std::vector<int> &shape, std::vector<int> &perm)
{
    /**
     * 1. merge shape[i] = 1;
     * 2. merge continue shape;
     */
    reduce_1_axes(shape, perm);
    fuse_consecutive_shape_perm_(shape, perm);
}

REGISTER_PLUGIN(Transpose2DPlugin, VERSION_NUMBER, BUILDERPLUGIN_TYPE);

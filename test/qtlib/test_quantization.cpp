#include "core/quantization.h"
#include "quantization_transform/ops/no_quantization.h"
#include "quantization_transform/ops/op_register.h"
#include "test_qtlib.h"

#include <vector>
using namespace aipubt;

void test_q()
{
    std::vector<float> scales = {1.5, 3.0};
    std::vector<int32_t> zps = {1, 12};
    // auto q = aipubt::Quantization(scales, zps);
    INFO() << "1";
    aipubt::Quantization q(scales, zps);
    INFO() << "2";
    auto qq = q;
    INFO() << "3";
    INFO() << qq.scales()[0];
    INFO() << qq.offsets()[0];
    auto t = aipubt::Quantization();
    float scale = 1.8;
    t.set_scale(scale);
    t.set_scale(scale);
    float s = t.scale();
    INFO() << s;

    t.set_scales(scales);
    INFO() << t.scales()[0] << ", " << t.scales()[1];

    auto x = qtlibn::OpRegistry::get_op_map_entry();
    for (auto optype_op : x)
    {
        auto op = optype_op.second;
        auto top = dynamic_cast<qtlibn::NoQuantization *>(op);
        if (top)
        {
            auto optype = optype_op.first;

            if (optype == OpType_Pooling3D || optype == OpType_Pooling)
            {
                continue;
            }
            INFO() << optype_op.first;
        }
    }
}
#include "internal_optype.h"
#include "core/optype.h"

#define INTERNAL_OPTYPE_REGISTER(name) \
    static aipubt::OPTypeRegister _register##name((#name) + 7, static_cast<aipubt::OpType>(name));

using namespace internal_ops;
INTERNAL_OPTYPE_REGISTER(OpType_Variance);
INTERNAL_OPTYPE_REGISTER(OpType_WordRsqrt);
INTERNAL_OPTYPE_REGISTER(OpType_NGammaQuant);
INTERNAL_OPTYPE_REGISTER(OpType_Norm);
INTERNAL_OPTYPE_REGISTER(OpType_NormGammaBeta);
INTERNAL_OPTYPE_REGISTER(OpType_SquareSum);
INTERNAL_OPTYPE_REGISTER(OpType_SurfaceSum);
INTERNAL_OPTYPE_REGISTER(OpType_SurfaceSubSum);
INTERNAL_OPTYPE_REGISTER(OpType_LayerSum);
INTERNAL_OPTYPE_REGISTER(OpType_LayerMean);
INTERNAL_OPTYPE_REGISTER(OpType_LayerSubMean);
INTERNAL_OPTYPE_REGISTER(OpType_RMSRsqrt);


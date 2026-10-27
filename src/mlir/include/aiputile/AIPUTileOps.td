#ifndef AIPUTILE_OPS
#define AIPUTILE_OPS
include "aiputile/AIPUTile.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIPUTile_Op<string name, list<Trait> traits = []>
    : Op<AIPUTile_Dialect, name, traits>;

// Tensor values are immutable, dense, static, identically shaped FP32 tensors.
// Buffer ownership and the output parameter ABI are introduced by tile lowering.
class AIPUTile_Binary<string name> : AIPUTile_Op<name, [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def AIPUTile_AddOp : AIPUTile_Binary<"add">;
def AIPUTile_MulOp : AIPUTile_Binary<"mul">;

def AIPUTile_FusedAddMulOp : AIPUTile_Op<"fused_add_mul", [Pure]> {
  let summary = "Compute (x + b) * a in logical tiles, without an intermediate tensor";
  let arguments = (ins AnyRankedTensor:$x, AnyRankedTensor:$b,
                       AnyRankedTensor:$a, I64Attr:$tile_size,
                       DefaultValuedAttr<BoolAttr, "true">:$add_on_lhs);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$x `,` $b `,` $a attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
#endif

#ifndef AIPUTILE_OPS
#define AIPUTILE_OPS
include "aiputile/AIPUTileTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIPUTile_Op<string name, list<Trait> traits = []>
    : Op<AIPUTile_Dialect, name, traits>;

def AIPUTile_FromTensorOp : AIPUTile_Op<"from_tensor", [Pure]> {
  let summary = "Expose a contiguous tensor as one immutable logical tile";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AIPUTile_TileType:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def AIPUTile_ToTensorOp : AIPUTile_Op<"to_tensor", [Pure]> {
  let summary = "Restore the external tensor shape from an immutable tile value";
  let arguments = (ins AIPUTile_TileType:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

class AIPUTile_Binary<string name> : AIPUTile_Op<name, [Pure]> {
  let arguments = (ins AIPUTile_TileType:$lhs, AIPUTile_TileType:$rhs);
  let results = (outs AIPUTile_TileType:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def AIPUTile_AddOp : AIPUTile_Binary<"add">;
def AIPUTile_MulOp : AIPUTile_Binary<"mul">;

def AIPUTile_TileLoopOp : AIPUTile_Op<"tile_loop",
    [Pure, IsolatedFromAbove, SingleBlock]> {
  let summary = "Partition inputs into equal-domain tiles and compute each tile";
  let description = [{
    Inputs and output have the same full logical length N. For tile size S,
    this loop evaluates ceil(N/min(S,N)) partitions. For each partition, the
    region receives one tile<min(S,N)xf32> per input, followed by two index
    arguments: offset and valid_length. Offset is the start of the partition;
    valid_length=min(min(S,N), N-offset) bounds all tile arithmetic and output.
    The region yields one tile of the partition capacity. It is isolated from
    external values and contains only tile-local Add/Mul operations and yield.
    The output is the ordered concatenation of valid partition lanes.
  }];
  let arguments = (ins Variadic<AIPUTile_TileType>:$inputs, I64Attr:$tile_size);
  let results = (outs AIPUTile_TileType:$output);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def AIPUTile_YieldOp : AIPUTile_Op<"yield",
    [Pure, Terminator, HasParent<"TileLoopOp">]> {
  let summary = "Yield the valid lanes produced for the current tile partition";
  let arguments = (ins AIPUTile_TileType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}
#endif

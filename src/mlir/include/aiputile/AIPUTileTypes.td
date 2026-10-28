#ifndef AIPUTILE_TYPES
#define AIPUTILE_TYPES
include "mlir/IR/AttrTypeBase.td"
include "aiputile/AIPUTile.td"

def AIPUTile_TileType : TypeDef<AIPUTile_Dialect, "Tile"> {
  let mnemonic = "tile";
  let summary = "Immutable contiguous logical tile with a bounded lane capacity";
  let description = [{
    A tile is a dense FP32 value with a static number of lanes. It does not
    imply a particular allocation or memory space. At a tile_loop boundary,
    each input tile is partitioned by the loop's common offset and valid_length.
    The final partition has fewer valid lanes than its capacity; invalid lanes
    may not be observed. Full input tiles are introduced by from_tensor and
    converted back only at the external tensor ABI boundary by to_tensor.
  }];
  let parameters = (ins "int64_t":$capacity, "Type":$elementType);
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}
#endif

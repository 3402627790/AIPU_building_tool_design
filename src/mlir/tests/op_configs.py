"""
Test configurations for all registered AIPUGC ops.

Each op entry specifies:
    aipuir_name  : layer type name accepted by the ``aipuir`` CLI
    shapes       : list of (label, input_shape_param_string) tuples
    dtypes       : list of (label, param_fragment_or_empty) tuples
    extra_params : optional static param fragment appended to every test
    category     : pytest marker name for selective runs
    multi_input  : whether the op expects >1 input with same shape

The ``aipuir`` tool auto-adapts unspecified parameters (output shape,
weight shape, padding, …) so we only need to provide values that we want
to control explicitly.
"""

# =========================================================================
# Dtype presets
# =========================================================================

# Integer dtypes (default IR)
from typing import Any


DTYPE_INT8 = ("int8", "")
DTYPE_INT16 = ("int16", "it=int16")
DTYPE_UINT8 = ("uint8", "it=uint8")

# Float dtypes (require float_ir=true)
DTYPE_FLOAT16 = ("float16", "float_ir=true")
DTYPE_BFLOAT16 = ("bfloat16", "float_ir=true; it=bfloat16")

INT_DTYPES = [DTYPE_INT8, DTYPE_INT16, DTYPE_UINT8]
FLOAT_DTYPES = [DTYPE_FLOAT16, DTYPE_BFLOAT16]
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES

# Binary ops require dtype for both inputs: it=<type>,<type>
BINARY_DTYPE_INT8 = ("int8", "it=int8,int8")
BINARY_DTYPE_INT16 = ("int16", "it=int16,int16")
BINARY_DTYPE_UINT8 = ("uint8", "it=uint8,uint8")
BINARY_DTYPE_FLOAT16 = ("float16", "float_ir=true; it=float16,float16")
BINARY_DTYPE_BFLOAT16 = ("bfloat16", "float_ir=true; it=bfloat16,bfloat16")

BINARY_INT_DTYPES = [BINARY_DTYPE_INT8, BINARY_DTYPE_INT16, BINARY_DTYPE_UINT8]
BINARY_FLOAT_DTYPES = [BINARY_DTYPE_FLOAT16, BINARY_DTYPE_BFLOAT16]
BINARY_ALL_DTYPES = BINARY_INT_DTYPES + BINARY_FLOAT_DTYPES

# =========================================================================
# Shape presets (NHWC layout unless noted)
# =========================================================================

SHAPES_4D = [
    ("4d_small",  "[1,4,4,4]"),
    ("4d_medium", "[1,10,10,8]"),
    ("4d_large",  "[1,32,32,16]"),
    ("4d_rect",   "[1,20,30,3]"),
    ("4d_batch",  "[2,8,8,4]"),
]

SHAPES_4D_MINIMAL = [
    ("4d_small",  "[1,4,4,4]"),
    ("4d_medium", "[1,10,10,8]"),
]

SHAPES_2D = [
    ("2d_small", "[1,64]"),
    ("2d_medium", "[4,128]"),
    ("2d_large", "[1,256]"),
]

SHAPES_GENERIC = [
    ("4d_small",  "[1,4,4,4]"),
    ("4d_medium", "[1,10,10,8]"),
    ("4d_large",  "[1,20,20,16]"),
]

SHAPES_3D = [
    ("3d_small", "[1,4,8]"),
    ("3d_medium", "[2,10,16]"),
]

SHAPES_BINARY = [
    ("4d_same_small",    "[[1,4,4,4],[1,4,4,4]]"),
    ("4d_same_medium",   "[[1,10,10,8],[1,10,10,8]]"),
    ("4d_same_large",    "[[1,20,20,16],[1,20,20,16]]"),
    ("4d_bcast_1d",      "[[1,4,4,4],[4]]"),
    ("4d_bcast_scalar",  "[[1,10,10,8],[1]]"),
]


# =========================================================================
# Op Configurations
# =========================================================================
# Each entry is a dict with keys: aipuir_name, shapes, dtypes,
# extra_params (optional), category, multi_input (optional).

OP_CONFIGS = [
    # =======================  Unary Ops  =======================
    {
        "aipuir_name": "Abs",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Negative",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Square",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Sqrt",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Rsqrt",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Cosine",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Exp",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Log",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },
    {
        "aipuir_name": "Quantize",
        "shapes": SHAPES_GENERIC,
        "dtypes": INT_DTYPES,
        "category": "unary",
    },

    # =======================  Binary Ops  =======================
    {
        "aipuir_name": "Add",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "Sub",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "Mul",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "Div",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "Eltwise",
        "shapes": [
            ("4d_small",  "[[1,4,4,4],[1,4,4,4]]"),
            ("4d_medium", "[[1,10,10,8],[1,10,10,8]]"),
            ("4d_large",  "[[1,20,20,16],[1,20,20,16]]"),
        ],
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "Pow",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },
    {
        "aipuir_name": "SquaredDifference",
        "shapes": SHAPES_BINARY,
        "dtypes": BINARY_ALL_DTYPES,
        "category": "binary",
    },

    # =======================  Convolution Ops  =======================
    {
        "aipuir_name": "Convolution",
        "shapes": [
            ("conv_small",  "[1,8,8,3]"),
            ("conv_medium", "[1,16,16,8]"),
            ("conv_large",  "[1,32,32,16]"),
            ("conv_rect",   "[1,20,30,3]"),
        ],
        "dtypes": ALL_DTYPES,
        "extra_params": "kx=3; ky=3",
        "category": "conv",
    },
    {
        "aipuir_name": "Convolution",
        "shapes": [
            ("conv_k5_medium", "[1,16,16,8]"),
        ],
        "dtypes": [DTYPE_INT8, DTYPE_FLOAT16],
        "extra_params": "kx=5; ky=5",
        "category": "conv",
    },
    {
        "aipuir_name": "Convolution",
        "shapes": [
            ("conv_k1_medium", "[1,16,16,16]"),
        ],
        "dtypes": [DTYPE_INT8, DTYPE_FLOAT16],
        "extra_params": "kx=1; ky=1",
        "category": "conv",
    },
    {
        "aipuir_name": "DepthwiseConv",
        "shapes": [
            ("dw_small",  "[1,8,8,4]"),
            ("dw_medium", "[1,16,16,8]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "conv",
    },

    # =======================  Pooling  =======================
    {
        "aipuir_name": "Pooling",
        "shapes": SHAPES_4D_MINIMAL + [
            ("pool_large", "[1,32,32,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "pool",
    },

    # =======================  Activation  =======================
    {
        "aipuir_name": "Activation",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "activation",
    },

    # =======================  Cast  =======================
    {
        "aipuir_name": "Cast",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "unary",
    },

    # =======================  Normalization Ops  =======================
    {
        "aipuir_name": "BatchNorm",
        "shapes": SHAPES_4D_MINIMAL + [
            ("bn_large", "[1,32,32,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "norm",
    },
    {
        "aipuir_name": "LayerNorm",
        "shapes": SHAPES_4D_MINIMAL + [
            ("ln_2d", "[1,128]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "norm",
    },
    {
        "aipuir_name": "RMSNorm",
        "shapes": SHAPES_4D_MINIMAL + [
            ("rmsn_2d", "[1,128]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "norm",
    },

    # =======================  Fully Connected  =======================
    {
        "aipuir_name": "FullyConnected",
        "shapes": SHAPES_2D,
        "dtypes": ALL_DTYPES,
        "category": "binary",
    },

    # =======================  MatMul  =======================
    {
        "aipuir_name": "MatMul",
        "shapes": [
            ("mm_small", "[1,4,8],[1,8,4]"),
            ("mm_medium", "[2,16,32],[2,32,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "binary",
    },

    # =======================  Softmax  =======================
    {
        "aipuir_name": "Softmax",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "reduce",
    },

    # =======================  Reduce  =======================
    {
        "aipuir_name": "Reduce",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "reduce",
    },

    # =======================  ArgMinMax  =======================
    {
        "aipuir_name": "ArgMinMax",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "reduce",
    },

    # =======================  Shape Manipulation  =======================
    {
        "aipuir_name": "Reshape",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Transpose",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Tile",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Concat",
        "shapes": [
            ("concat_small", "[1,4,4,8],[1,4,4,8]"),
            ("concat_large", "[1,20,20,16],[1,20,20,16]"),
        ],
        "dtypes": [
            ("concat_int8", "int8,int8"),
            ("concat_uint16", "uint8,uint8"),
            ("concat_float16", "float16,float16"),
            ("concat_bfloat16", "bfloat16,bfloat16"),
        ],
        "category": "shape",
    },
    {
        "aipuir_name": "Split",
        "shapes": [
            ("split_small",  "[1,4,4,8]"),
            ("split_medium", "[1,10,10,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Slice",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Pad",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Crop",
        "shapes": [
            ("crop_medium", "[1,10,10,8]"),
            ("crop_large",  "[1,20,20,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "Gather",
        "shapes": SHAPES_GENERIC,
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },
    {
        "aipuir_name": "ChannelShuffle",
        "shapes": [
            ("cs_small",  "[1,4,4,8]"),
            ("cs_medium", "[1,10,10,16]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "shape",
    },

    # =======================  Spatial Rearrangement  =======================
    {
        "aipuir_name": "DepthToSpace",
        "shapes": [
            ("d2s_small",  "[1,4,4,16]"),
            ("d2s_medium", "[1,8,8,32]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },
    {
        "aipuir_name": "SpaceToDepth",
        "shapes": [
            ("s2d_small",  "[1,4,4,4]"),
            ("s2d_medium", "[1,8,8,8]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },
    {
        "aipuir_name": "BatchToSpace",
        "shapes": [
            ("b2s_small",  "[4,4,4,4]"),
            ("b2s_medium", "[4,8,8,8]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },
    {
        "aipuir_name": "SpaceToBatch",
        "shapes": [
            ("s2b_small",  "[1,4,4,4]"),
            ("s2b_medium", "[1,8,8,8]"),
        ],
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },

    # =======================  Image / Resize  =======================
    {
        "aipuir_name": "Resize",
        "shapes": SHAPES_4D_MINIMAL,
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },
    {
        "aipuir_name": "GridSample",
        "shapes": SHAPES_4D_MINIMAL,
        "dtypes": ALL_DTYPES,
        "category": "spatial",
    },

    # =======================  Sequence Ops  =======================
    {
        "aipuir_name": "ReverseSequence",
        "shapes": SHAPES_4D_MINIMAL,
        "dtypes": ALL_DTYPES,
        "category": "sequence",
    },
    {
        "aipuir_name": "BasicLSTM",
        "shapes": SHAPES_3D,
        "dtypes": ALL_DTYPES,
        "category": "sequence",
    },
]


def generate_test_cases():
    """
    Yield ``(test_id, op_name, param_string, category)`` tuples for every
    (op x shape x dtype) combination.
    """
    for cfg in OP_CONFIGS:
        op_name = cfg["aipuir_name"]
        extra = cfg.get("extra_params", "")
        category = cfg.get("category", "")

        for shape_label, shape_val in cfg["shapes"]:
            for dtype_label, dtype_param in cfg["dtypes"]:
                parts = [f"is={shape_val}"]
                if dtype_param:
                    parts.append(dtype_param)
                if extra:
                    parts.append(extra)
                param_str = "; ".join(parts)

                tid = f"{op_name}-{shape_label}-{dtype_label}"

                yield tid, op_name, param_str, category


ALL_TEST_CASES = list(generate_test_cases())

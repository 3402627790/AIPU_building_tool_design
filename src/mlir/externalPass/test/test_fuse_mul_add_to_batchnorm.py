import os
import ctypes
import unittest
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

BUILD_DIR = "/project/ai/scratch01/qidson01/code/AIPU_building_tool_design/src/mlir/build"
DIALECT_SO = f"{BUILD_DIR}/libAIPUMLIR.so"
PLUGIN_SO  = f"{BUILD_DIR}/externalPass/libexternalPass.so"


_dialect_lib = ctypes.CDLL(DIALECT_SO, mode=ctypes.RTLD_GLOBAL)
_dialect_lib.aipugcRegisterDialects.argtypes = [ctypes.c_void_p]
_dialect_lib.aipugcRegisterDialects.restype = None
ctypes.CDLL(PLUGIN_SO, mode=ctypes.RTLD_GLOBAL)

MUL_ADD_MLIR = """
module {
  aipugc.func @model(%arg0: tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>) -> tensor<2x256x!aipugc.quant<i8:f32, 1.000000>> {
    %0 = "aipugc.constant"() {value = dense<0> : tensor<256xi8>} : () -> tensor<256x!aipugc.quant<i8:f32, 1.000000>>
    %1 = "aipugc.constant"() {value = dense<0> : tensor<256xi8>} : () -> tensor<256x!aipugc.quant<i8:f32, 1.000000>>
    %2 = "aipugc.mul"(%arg0, %0) {scale_type = "uint8", scale_value = 128 : i32, shift_type = "int8", shift_value = 29 : i32, with_activation = "NONE"} : (tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>, tensor<256x!aipugc.quant<i8:f32, 1.000000>>) -> tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>
    %3 = "aipugc.add"(%2, %1) {scale_type = ["uint8", "uint16", "uint16"], scale_value = [188 : i32, 32767 : i32, 256 : i32], shift_type = "int8", shift_value = 29 : i32, with_activation = "NONE"} : (tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>, tensor<256x!aipugc.quant<i8:f32, 1.000000>>) -> tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>
    aipugc.return %3 : tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>
  }
}
"""

PIPELINE = "builtin.module(aipugc.func(fuse-mul-add-to-batchnorm,cse))"


def apply_pass(mlir_code: str) -> str:
    with Context() as ctx:
        # Register aipugc dialect into the current Context
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        raw_ptr = ctypes.pythonapi.PyCapsule_GetPointer(
            ctx._CAPIPtr, b"mlir.ir.Context._CAPIPtr")
        _dialect_lib.aipugcRegisterDialects(raw_ptr)
        module = Module.parse(mlir_code)
        pm = PassManager.parse(PIPELINE)
        pm.run(module.operation)
        result = str(module)
        print("\n=== Pass output ===")
        print(result)
        print("===================\n")
        return result


class TestFuseMulAddToBatchNorm(unittest.TestCase):

    def setUp(self):
        self.result = apply_pass(MUL_ADD_MLIR)

    def test_batchnorm_present(self):
        self.assertIn("aipugc.BatchNorm", self.result)

    def test_mul_removed(self):
        self.assertNotIn('"aipugc.mul"', self.result)

    def test_add_removed(self):
        self.assertNotIn('"aipugc.add"', self.result)

    def test_output_type_preserved(self):
        self.assertIn("tensor<2x256x!aipugc.quant<i8:f32, 1.000000>>", self.result)

    def test_dce_merges_duplicate_constants(self):
        self.assertEqual(self.result.count("aipugc.constant"), 1)


if __name__ == "__main__":
    TestFuseMulAddToBatchNorm().setUp()
    unittest.main()
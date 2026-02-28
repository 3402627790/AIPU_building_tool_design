import numpy as np
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

def compile_mlir(mlir_module_str):
    with Context():
        module = Module.parse(mlir_module_str)
        module = apply_gpu_pipeline(module)
    return module

def apply_gpu_pipeline(module):
    pm = PassManager()
    
    pm.add("convert-gpu-to-spirv")
    
    # pm.add("reconcile-unrealized-casts")
    # pm.add("canonicalize")
    
    pm.run(module.operation)
    print(module)
    return module

MLIR_CODE = """
module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024 : i32, 1024 : i32, 64 : i32], min_subgroup_size = 1, max_subgroup_size = 32>>} {
  func.func @Add(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    gpu.launch_func  @Add_kernel::@Add_kernel blocks in (%c2, %c1, %c1) threads in (%c2, %c1, %c1)  args(%c0 : index, %c0 : index, %arg0 : memref<2x2xf32>, %arg1 : memref<2x2xf32>, %alloc : memref<2x2xf32>)
    return %alloc : memref<2x2xf32>
  }
  gpu.module @Add_kernel {
    gpu.func @Add_kernel(%arg0: index, %arg1: index, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>, %arg4: memref<2x2xf32>) 
    kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>}
  {
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.addi %arg1, %thread_id_x : index
      %2 = memref.load %arg2[%0, %1] : memref<2x2xf32>
      %3 = memref.load %arg3[%0, %1] : memref<2x2xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg4[%0, %1] : memref<2x2xf32>
      gpu.return
    }
  }
}

"""

mlir_module = compile_mlir(MLIR_CODE)
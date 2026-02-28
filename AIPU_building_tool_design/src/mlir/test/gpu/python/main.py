import ctypes
import numpy as np
import cuda.cuda as cu  # type: ignore
import cuda.cudart as cudart  # type: ignore
import cuda.nvrtc as nvrtc  # type: ignore
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
import subprocess


# Error handling utilities
def _cudaGetErrorEnum(error):
    if isinstance(error, cu.CUresult):
        err, name = cu.cuGetErrorName(error)
        return name if err == cu.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def compile_mlir_to_ptx(mlir_module_str: str, chip_type="sm_70"):
    """Compiles MLIR module string to PTX code."""
    with Context():
        # Parse the input module
        module = Module.parse(mlir_module_str)

        # Apply GPU compilation pipeline
        module, gpu_module = apply_gpu_pipeline(module, chip_type)

        # Generate PTX from the GPU module
        ptx = generate_ptx(str(gpu_module), chip_type)

    return ptx


def apply_gpu_pipeline(module, chip_type="sm_70"):
    """Applies the GPU compilation pipeline to the MLIR module."""
    pm = PassManager()
    pm.enable_ir_printing(print_after_change=True)
    pm.add("canonicalize")
    pm.add(
        "one-shot-bufferize{ bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map }"
    )
    pm.add("canonicalize")
    pm.add("convert-linalg-to-affine-loops")
    pm.add("func.func(affine-loop-invariant-code-motion)")
    pm.add("func.func(convert-affine-for-to-gpu)")
    pm.add("gpu-kernel-outlining")
    pm.add("lower-affine")
    pm.add("gpu-decompose-memrefs")
    pm.add("expand-strided-metadata")
    pm.add("normalize-memrefs")
    pm.add(
        "gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv })"
    )
    pm.add(f"nvvm-attach-target{{chip={chip_type} features=+ptx80 O=3}}")
    pm.add("convert-nvvm-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.add("gpu-to-llvm { use-bare-pointers-for-host use-bare-pointers-for-kernels }")
    pm.run(module.operation)

    gpu_module = extract_gpu_module(module)

    return module, gpu_module


def extract_gpu_module(module: Module) -> Module:
    """Extracts the GPU module from a transformed MLIR module."""
    try:
        main_func_op = module.operation.regions[0].blocks[0].operations[2]  # the third op, because  %0 = tensor.empty(), %1 = linalg.generic
        gpu_module_op = main_func_op.regions[0].blocks[0].operations[0]
        gpu_module = Module.parse(str(gpu_module_op))
        return gpu_module
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to extract GPU module: {e}") from e


def generate_ptx(gpu_module_str, chip_type="sm_70"):
    """Generates PTX from an MLIR GPU module string."""
    llvm_ir_result = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir", "-"],
        input=gpu_module_str,
        capture_output=True,
        text=True,
    )

    if llvm_ir_result.returncode != 0:
        print("Error generating LLVM IR:")
        print(llvm_ir_result.stderr)
        return None

    llvm_ir = llvm_ir_result.stdout

    # Then convert LLVM IR to PTX
    ptx_result = subprocess.run(
        ["llc", "-march=nvptx64", f"-mcpu={chip_type}", "-"],
        input=llvm_ir,
        capture_output=True,
        text=True,
    )

    if ptx_result.returncode != 0:
        print("Error generating PTX:")
        print(ptx_result.stderr)
        return None

    return ptx_result.stdout


# CUDA memory and execution functions
def setup_cuda(device_id=0):
    """Initialize CUDA and create a context."""
    print("Initializing CUDA...")
    checkCudaErrors(cu.cuInit(0))
    device = checkCudaErrors(cu.cuDeviceGet(device_id))
    context = checkCudaErrors(cu.cuCtxCreate(0, device))
    print(f"CUDA context created on device {device_id}.")
    return context


def cleanup_cuda(context):
    """Destroy the CUDA context."""
    if context:
        print("Destroying CUDA context...")
        checkCudaErrors(cu.cuCtxDestroy(context))
        print("CUDA context destroyed.")


def allocate_device_memory(size_bytes):
    """Allocate memory on the GPU."""
    return checkCudaErrors(cu.cuMemAlloc(size_bytes))


def free_device_memory(device_ptr):
    """Free memory on the GPU."""
    if device_ptr:
        checkCudaErrors(cu.cuMemFree(device_ptr))


def copy_host_to_device(host_array, device_ptr):
    """Copy data from host to device."""
    if not host_array.flags.c_contiguous:
        host_array = np.ascontiguousarray(host_array)
    checkCudaErrors(
        cu.cuMemcpyHtoD(device_ptr, host_array.ctypes.data, host_array.nbytes)
    )


def copy_device_to_host(device_ptr, host_array):
    """Copy data from device to host."""
    checkCudaErrors(
        cu.cuMemcpyDtoH(host_array.ctypes.data, device_ptr, host_array.nbytes)
    )


def run_kernel(
    ptx_code,
    kernel_name,
    args,
    arg_types,
    grid_dims,
    block_dims,
):
    """Run a PTX kernel."""
    module = checkCudaErrors(cu.cuModuleLoadData(ptx_code.encode("utf-8")))

    kernel_func = checkCudaErrors(
        cu.cuModuleGetFunction(module, kernel_name.encode("utf-8"))
    )

    kernel_args = (tuple(args), tuple(arg_types))

    checkCudaErrors(
        cu.cuLaunchKernel(
            kernel_func,
            grid_dims[0],
            grid_dims[1],
            grid_dims[2],
            block_dims[0],
            block_dims[1],
            block_dims[2],
            0,  # shared memory bytes
            0,  # stream
            kernel_args,  # kernel args
            0,  # extra
        )
    )

    checkCudaErrors(cu.cuCtxSynchronize())

    checkCudaErrors(cu.cuModuleUnload(module))


# MLIR code for addition operation
MLIR_CODE = """
module {
  func.func @Add(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = tensor.empty() : tensor<10x10xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<10x10xf32>, tensor<10x10xf32>) outs(%0 : tensor<10x10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
  }
}
"""

# Prepare two input matrices
size = 10
input_data1 = np.random.randn(size, size).astype(np.float32)
input_data2 = np.random.randn(size, size).astype(np.float32)

# Expected output is the sum of the two matrices
expected_output = input_data1 + input_data2

# Step 1: Compile MLIR to PTX
print("Compiling MLIR to PTX...")
ptx_code = compile_mlir_to_ptx(MLIR_CODE)

if not ptx_code:
    raise RuntimeError("PTX compilation failed.")

# Step 2: Initialize CUDA
cuda_context = setup_cuda()

try:
    # Allocate three device memories: two inputs, one output
    d_input1 = allocate_device_memory(input_data1.nbytes)
    d_input2 = allocate_device_memory(input_data2.nbytes)
    output_data = np.zeros((size, size), dtype=np.float32)
    d_output = allocate_device_memory(output_data.nbytes)

    # Copy input data to device
    copy_host_to_device(input_data1, d_input1)
    copy_host_to_device(input_data2, d_input2)

    # Run kernel
    grid_dims = (size, 1, 1)  # One thread block per row
    block_dims = (size, 1, 1)  # One thread per column

    # Modified parameter list: added second input pointer
    args = [
        0,  # Grid dimension offset
        0,  # Block dimension offset
        d_input1,  # First input pointer
        d_input2,  # Second input pointer
        d_output,  # Output pointer
    ]
    arg_types = [
        ctypes.c_int,
        ctypes.c_int,
        None,  # Pointer type
        None,  # Pointer type
        None   # Pointer type
    ]

    print("Running kernel on GPU...")
    run_kernel(
        ptx_code,
        "Add_kernel",
        args,
        arg_types,
        grid_dims,
        block_dims,
    )

    # Copy results back to host
    copy_device_to_host(d_output, output_data)

    # Verify results (check addition result)
    print("Verifying results...")
    np.testing.assert_allclose(output_data, expected_output, rtol=1e-5)
    print("Success! Results verified.")

finally:
    # Clean up resources
    free_device_memory(d_input1)
    free_device_memory(d_input2)
    free_device_memory(d_output)
    cleanup_cuda(cuda_context)
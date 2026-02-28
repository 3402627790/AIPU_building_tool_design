#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>

struct MemRefDescriptor {
    void* allocated;
    void* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" {
    void _mlir_ciface_Add(MemRefDescriptor* result, MemRefDescriptor* input1, MemRefDescriptor* input2);
    
    void* malloc(size_t size);
    void free(void* ptr);
}

struct AlignedDeleter {
    void operator()(void* ptr) const { 
        if (ptr) std::free(ptr); 
    }
};

using AlignedFloatPtr = std::unique_ptr<float[], AlignedDeleter>;

AlignedFloatPtr createAlignedFloatArray(size_t count) {
    void* ptr = aligned_alloc(64, count * sizeof(float));
    if (!ptr) {
        throw std::bad_alloc();
    }
    return AlignedFloatPtr(static_cast<float*>(ptr));
}

MemRefDescriptor createMemRef2D(float* data, int64_t dim0, int64_t dim1) {
    MemRefDescriptor desc;
    desc.allocated = data;
    desc.aligned = data;
    desc.offset = 0;
    desc.sizes[0] = dim0;
    desc.sizes[1] = dim1;
    desc.strides[0] = dim1;
    desc.strides[1] = 1;
    return desc;
}

void printMatrix(float* data, int64_t rows, int64_t cols, const char* name) {
    std::cout << name << " = [" << std::endl;
    for (int64_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (int64_t j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main() {
    const int64_t dim0 = 2, dim1 = 2;
    const size_t total_size = dim0 * dim1;
    
    AlignedFloatPtr input1_data = createAlignedFloatArray(total_size);
    input1_data[0] = 1.0f; input1_data[1] = 2.0f;
    input1_data[2] = 3.0f; input1_data[3] = 4.0f;
    
    AlignedFloatPtr input2_data = createAlignedFloatArray(total_size);
    input2_data[0] = 5.0f; input2_data[1] = 6.0f;
    input2_data[2] = 7.0f; input2_data[3] = 8.0f;
    
    MemRefDescriptor result_desc{};
    
    MemRefDescriptor input1_desc = createMemRef2D(input1_data.get(), dim0, dim1);
    MemRefDescriptor input2_desc = createMemRef2D(input2_data.get(), dim0, dim1);
    
    std::cout << "Input matrices:" << std::endl;
    printMatrix(input1_data.get(), dim0, dim1, "input1");
    printMatrix(input2_data.get(), dim0, dim1, "input2");
    
    std::cout << "Calling _mlir_ciface_Add..." << std::endl;
    _mlir_ciface_Add(&result_desc, &input1_desc, &input2_desc);
    
    std::cout << "Result:" << std::endl;
    float* result_data = static_cast<float*>(result_desc.aligned);
    printMatrix(result_data, dim0, dim1, "result");
    
    if (result_desc.allocated) {
        std::free(result_desc.allocated);
    }
    
    return 0;
}
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

struct MemRefDescriptor {
    void* allocated;  // Base address of the actual memory allocation
    void* aligned;    // Aligned pointer to the start of effective usable data
    long offset;      // Byte offset from aligned pointer to first element (allocated -> aligned)
    long sizes[2];    // Number of elements in each dimension
    long strides[2];  // Element-wise stride for each dimension
};

extern "C" MemRefDescriptor Add(
    void* a_allocated, void* a_aligned, long a_offset, long a_size0, long a_size1, long a_stride0, long a_stride1,
    void* b_allocated, void* b_aligned, long b_offset, long b_size0, long b_size1, long b_stride0, long b_stride1
);

MemRefDescriptor AddFunc(MemRefDescriptor a, MemRefDescriptor b)
{
    return Add(
        a.allocated, a.aligned, a.offset, a.sizes[0], a.sizes[1], a.strides[0], a.strides[1],
        b.allocated, b.aligned, b.offset, b.sizes[0], b.sizes[1], b.strides[0], b.strides[1]
    );
}

void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
    #endif
    return ptr;
}


void aligned_free(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

void print_matrix(const char* name, float* data, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int rows = 2;
    const int cols = 2;
    const size_t buffer_size = rows * cols * sizeof(float);
    const size_t alignment = 64;

    float* a_data = static_cast<float*>(aligned_alloc(alignment, buffer_size));
    a_data[0] = 1.0f; a_data[1] = 2.0f;
    a_data[2] = 3.0f; a_data[3] = 4.0f;
    print_matrix("Matrix A", a_data, rows, cols);

    float* b_data = static_cast<float*>(aligned_alloc(alignment, buffer_size));
    b_data[0] = 0.5f; b_data[1] = 1.0f;
    b_data[2] = 1.5f; b_data[3] = 2.0f;
    print_matrix("Matrix B", b_data, rows, cols);

    MemRefDescriptor a_desc = {
        a_data, a_data, 0,
        {rows, cols},
        {cols, 1}
    };

    MemRefDescriptor b_desc = {
        b_data, b_data, 0,
        {rows, cols},
        {cols, 1}
    };

    MemRefDescriptor result = AddFunc(a_desc, b_desc);

    float* result_data = static_cast<float*>(result.aligned);
    print_matrix("Result Matrix (A + B)", result_data, rows, cols);

    aligned_free(a_data);
    aligned_free(b_data);
    free(result.allocated);

    return 0;
}

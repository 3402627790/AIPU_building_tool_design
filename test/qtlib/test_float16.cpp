#include "test_qtlib.h"

void test_float16(void)
{
    float f = 1.839566668701172f;
    half_float::half f16 = static_cast<half_float::half>(f);
    std::cout << "float32 = " << f << ", float16 = " << f16;
}
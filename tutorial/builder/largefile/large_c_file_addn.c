#include <aipu.h>
__entry void add_n(int32_t n, uint32_t size,
                      uint8_t *input, uint8_t *output)
{
    for (int i = 0; i < size; i++)
    {
        output[i]=input[i]+n;
    }
}
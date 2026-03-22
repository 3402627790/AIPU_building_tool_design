#include "asm/asm.h"
#include "vm/asmtranscode.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>

using namespace aipubt;

uint64_t get_time_in_us()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * 1000000 + now.tv_usec;
}
__attribute__((optimize("O0"))) unsigned fbn(unsigned n)
{
    unsigned ret = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        ret += i;
    }
    return ret;
}
int main()
{
    std::vector<Instruction> codes =
        {
            "ADD R2,R30,0"_asm,
            "BR R1"_asm,
            "MOVIL R1,10"_asm,
            "BR R30"_asm,
            "MOVIL R1, 100"_asm,
            "MOVIL R2, 0x100"_asm,
            "BR R30"_asm,
        };

    aipubt::TransCode tc(codes);
    std::cout << "test" << std::endl;
    auto ff = tc.get_code(0);
    TecStatus status = {0};
    status.tec = &tc;
    // int n = 10;
    int r = 1;
    // int result = 0;
    // auto t0 = get_time_in_us();
    for (auto i = 0; i < r; ++i)

    {
        status.RReg[0] = 0;
        status.RReg[1] = 0x50;
        status.RReg[30] = 0xfffffff0;

        ff(&status);
        if (status.RReg[1] != 0x50 || status.RReg[2] != 0x100 || status.RReg[30] != 0xfffffff0)
        {
            std::cout << "Error!!!\n";
        }
    }
    // auto t1 = get_time_in_us();
    // for (auto i = 0; i < r; ++i)
    //     result = fbn(n);
    // auto t2 = get_time_in_us();

    // std::cout << "exited," << std::endl
    //           << "JIT cost " << (double)(t1 - t0) / 1000 << "ms, result=" << status.RReg[0] << std::endl
    //           << "NATIVE cost " << (double)(t2 - t1) / 1000 << "ms, result=" << result << std::endl
    //           << std::endl
    //           << std::endl;

    return 0;
}
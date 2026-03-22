//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang 
 * @Date: 2021-08-09 09:38:52 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 15:05:44
 */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
extern "C" int aipuoas(int argc, char **argv);
extern "C" int aipuold(int argc, char **argv);

void segment_fault_handler()
{
    printf("exit %d-----\n", 100);
}

extern "C"
{
    int btaipuas(int argc, char **argv)
    {
        int m_ret = -1;
        pid_t pid = fork();
        if (pid == 0)
        {
            aipuoas(argc, argv);
            std::vector<const char *> ArgVec;
        }
        else if (pid > 0)
        {
            // parent process
            wait(&m_ret);
            int es;
            if (WIFEXITED(m_ret))
            {
                es = WEXITSTATUS(m_ret);
                printf("Exit status was %d\n", es);
            }
            printf("%d\n", m_ret);
            return es;
        }
        else
        {
            // fork failed
            // std::cerr << "fork() failed!" << std::endl;
        }
        printf("host %d\n", m_ret);

        return m_ret;
    }

    int btaipuld(int argc, char **argv)
    {
        int m_ret = -1;
        pid_t pid = fork();
        if (pid == 0)
        {
            exit(aipuold(argc, argv));
        }
        else if (pid > 0)
        {
            // parent process
            wait(&m_ret);
            int es;
            if (WIFEXITED(m_ret))
            {
                es = WEXITSTATUS(m_ret);
            }
            return es;
        }
        return m_ret;
    }
};

int main(int argc, char **argv)
{
    std::vector<const char *> ArgVec;

    ArgVec.emplace_back("aipuoas");
    ArgVec.emplace_back("test.s");
    ArgVec.emplace_back("--arch=aipu");
    ArgVec.emplace_back("--filetype-gb=obj");
    ArgVec.emplace_back("--mcpu-gb=X2_1204");
    ArgVec.emplace_back("-o");
    ArgVec.emplace_back("test.o");
    int Argc = ArgVec.size();
    const char **Argv = ArgVec.data();
    btaipuas(ArgVec.size(), const_cast<char **>(Argv));

    ArgVec.clear();
    ArgVec.emplace_back("aipuold");
    ArgVec.emplace_back("test.o");
    ArgVec.emplace_back("rpadd_int8.o");
    ArgVec.emplace_back("rpadd.o");
    ArgVec.emplace_back("-static");
    ArgVec.emplace_back("--Tbss=0xf8000000");
    ArgVec.emplace_back("--Ttext=0x20000");
    ArgVec.emplace_back("-o");
    ArgVec.emplace_back("main.elf");

    Argc = ArgVec.size();
    Argv = ArgVec.data();
    btaipuld(Argc, const_cast<char **>(Argv));

    return 0;
}
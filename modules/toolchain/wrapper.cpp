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
 * @Last Modified time: 2023-03-22 10:52:55
 */
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utils/utils.h>
#include <vector>
extern "C" int aipuoas(int argc, char **argv);
extern "C" int aipuold(int argc, char **argv);
extern "C" int aipuocc(int argc, char **argv);

extern "C"
{
    int btaipuas(int argc, char **argv)
    {
#ifdef NO_FORK
        return aipuoas(argc, argv);
#endif
        int m_ret = -1;
        pid_t pid = fork();
        if (pid == 0)
        {
            utils::clear_atexit();
            exit(aipuoas(argc, argv));
        }
        else
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
        return 0;
    }

    int btaipuld(int argc, char **argv)
    {
#ifdef NO_FORK
        return aipuold(argc, argv);
#endif
        int m_ret = -1;
        pid_t pid = fork();
        if (pid == 0)
        {
            utils::clear_atexit();
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
        return 0;
    }

    int btaipucc(int argc, char **argv)
    {
#ifdef NO_FORK
        return aipuocc(argc, argv);
#endif
        int m_ret = -1;
        pid_t pid = fork();
        if (pid == 0)
        {
            utils::clear_atexit();
            exit(aipuocc(argc, argv));
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
        return 0;
    }
};

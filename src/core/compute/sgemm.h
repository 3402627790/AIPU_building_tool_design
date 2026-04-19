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
 * @Date: 2020-02-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 16:40:41
 */
#ifndef CORE_COMPUTE_SGEMM_H
#define CORE_COMPUTE_SGEMM_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"

namespace aipubt
{
template <typename T>
void gemm_nn(int M, int N, int K, T ALPHA,
             T *A, int lda,
             T *B, int ldb,
             T *C, int ldc)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            T A_PART = ALPHA * A[i * lda + k];
            for (int j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

template <typename T>
void gemm_nt(int M, int N, int K, T ALPHA,
             T *A, int lda,
             T *B, int ldb,
             T *C, int ldc)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            T sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

template <typename T>
void gemm_tn(int M, int N, int K, T ALPHA,
             T *A, int lda,
             T *B, int ldb,
             T *C, int ldc)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            T A_PART = ALPHA * A[k * lda + i];
            for (int j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

template <typename T>
void gemm_tt(int M, int N, int K, T ALPHA,
             T *A, int lda,
             T *B, int ldb,
             T *C, int ldc)
{
#pragma omp parallel for
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            T sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

template <typename T>
void gemm_cpu(int TA, int TB, int M, int N, int K, T ALPHA,
              T *A, int lda,
              T *B, int ldb,
              T BETA,
              T *C, int ldc)
{
    if (BETA == 0)
    {
        memset(C, 0, sizeof(T) * M * N);
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                C[i * ldc + j] *= BETA;
            }
        }
    }

    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

MULTITYPES_COMPUTE_OP(gemm_cpu)
};     // namespace aipubt
#endif //!CORE_COMPUTE_SGEMM_H

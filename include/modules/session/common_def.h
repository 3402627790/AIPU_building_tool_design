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
 * @Author: Alex Zheng
 * @Date: 2026-04-14 14:24:30
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2026-04-14 14:15:49
 */
#ifndef __SESSION_COMMON_DEF_H__
#define __SESSION_COMMON_DEF_H__

/**
 * @brief Quantization mode of graph compilation
 * ORIGIN quantization formula: f = (Q - zp) * s (currently only support this formula)
 * Compass quantization formula: f = (Q + zp) / s
 */
typedef enum CpsQuantMode
{
    /**
     * quantization convert: `zp_cps = -zp_origin, scale_cps = 1 / scale_origin`
     * filter data quantization is always symmetric
     */
    CPS_QUANT_FASYM,
    /**
     * 8bit will be translated to 16bit data type
     * quantization convert: `zp_cps = 0, scale_cps = 1 / scale_origin`
     * filter data quantization is symmetric, and filter data is
     * `filter_cps = filter_origin - zp_origin`
     */
    CPS_QUANT_FSYM,

    CPS_QUANT_MAX,
} CpsQuantMode;

/**
 * @brief Priority of graph execution
 */
typedef enum CpsRunPriority
{
    CPS_RUN_PRIORITY_LOW,
    CPS_RUN_PRIORITY_HIGH,
    CPS_RUN_PRIORITY_MAX,
} CpsRunPriority;

/**
 * @brief Return code of API functions.
 *        - `SS_NO_ERROR` equals to (int32_t)0, represents success of calling
 *        - Other positive integers represent different meaning of errors
 */
typedef enum SessionResult
{
    SS_NO_ERROR = 0,
    SS_ERROR_INVALID_RESULT_STATUS,
    SS_ERROR_GENERAL_FAIL,
    SS_ERROR_NULLPTR,
    SS_ERROR_INVALID_ARG,
    SS_ERROR_INVALID_LAYERLIB,
    SS_ERROR_INVALID_NODE,
    SS_ERROR_CREATE_PATH_FAIL,
    SS_ERROR_PATH_NOT_FOUND,
    SS_ERROR_FILE_NOT_FOUND,
    SS_ERROR_INVALID_FILE_STATE,
    SS_ERROR_INVALID_FILE_SIZE,

    SS_ERROR_CREATE_WORKLOAD_FAIL,
    SS_ERROR_WORKLOAD_NOT_EXIST,
    SS_ERROR_INVALID_CACHE_FILE_INFO,
    SS_ERROR_INPUTS_NUM_NOT_MATCH,
    SS_ERROR_OUTPUTS_NUM_NOT_MATCH,
    SS_ERROR_OUTPUT_INSUFFICIENT,
    SS_ERROR_TENSOR_SIZE_NOT_MATCH,
    SS_ERROR_GRAPH_NOT_COMPILED,
    SS_ERROR_GRAPH_COMPILED_SIZE_NOT_MATCH,

    SS_ERROR_ADD_LAYER_FAIL,
    SS_ERROR_OPERATION_NOT_SUPPORT,
    SS_ERROR_PRE_PASS_FAIL,
    SS_ERROR_POST_PASS_FAIL,
    SS_ERROR_CREATE_CPS_GRAPH_FAIL,
    SS_ERROR_QUANTIZE_FAIL,
    SS_ERROR_GSIM_FAIL,
    SS_ERROR_GRAPH_CHECK_FAIL,
    SS_ERROR_BUILD_AIPU_FAIL,
    SS_ERROR_BAD_HASH,
    SS_ERROR_WRITE_CACHE_FAIL,
    SS_ERROR_CONVERT_NHWC_FAIL,
    SS_ERROR_OPTIMIZE_FAIL,
    SS_ERROR_GRAPH_NOT_FOUND,

    SS_ERROR_GRAPH_NOT_FINISHED,
    SS_ERROR_FINISH_EXECUTE_FAIL,
    SS_ERROR_FLUSH_EXECUTE_FAIL,

    SS_ERROR_INVALID_DEVICE,
    SS_ERROR_INVALID_TARGET,

    SS_ERROR_DMA_BUFFER_INVALID,
    SS_ERROR_DMA_BUFFER_ALLOC_FAIL,
    SS_ERROR_DMA_BUFFER_FREE_FAIL,
    SS_ERROR_DMA_BUFFER_SPECIFY_FAIL,
    SS_ERROR_DMA_BUFFER_NOT_FOUND,

    SS_ERROR_UMD_START,
    SS_ERROR_UMD_NULL_PTR,
    SS_ERROR_UMD_INVALID_CTX,
    SS_ERROR_UMD_OPEN_DEV_FAIL,
    SS_ERROR_UMD_DEV_ABNORMAL,
    SS_ERROR_UMD_DEINIT_FAIL,
    SS_ERROR_UMD_INVALID_CONFIG,
    SS_ERROR_UMD_UNKNOWN_BIN,
    SS_ERROR_UMD_GVERSION_UNSUPPORTED,
    SS_ERROR_UMD_INVALID_GBIN,
    SS_ERROR_UMD_TARGET_NOT_FOUND,
    SS_ERROR_UMD_INVALID_GRAPH_ID,
    SS_ERROR_UMD_OPEN_FILE_FAIL,
    SS_ERROR_UMD_MAP_FILE_FAIL,
    SS_ERROR_UMD_READ_FILE_FAIL,
    SS_ERROR_UMD_WRITE_FILE_FAIL,
    SS_ERROR_UMD_INVALID_JOB_ID,
    SS_ERROR_UMD_JOB_EXCEPTION,
    SS_ERROR_UMD_JOB_TIMEOUT,
    SS_ERROR_UMD_OP_NOT_SUPPORTED,
    SS_ERROR_UMD_INVALID_OP,
    SS_ERROR_UMD_INVALID_SIZE,
    SS_ERROR_UMD_BUF_ALLOC_FAIL,
    SS_ERROR_UMD_BUF_FREE_FAIL,
    SS_ERROR_UMD_INVALID_CORE_ID,
    SS_ERROR_UMD_RESERVE_SRAM_FAIL,
    SS_ERROR_UMD_INVALID_TENSOR_ID,
    SS_ERROR_UMD_INVALID_CLUSTER_ID,
    SS_ERROR_UMD_INVALID_PARTITION_ID,
    SS_ERROR_UMD_PRINTF_FAIL,
    SS_ERROR_UMD_INVALID_TENSOR_TYPE,
    SS_ERROR_UMD_INVALID_GM,
    SS_ERROR_UMD_INVALID_SEGMMU,
    SS_ERROR_UMD_INVALID_QOS,
    SS_ERROR_UMD_INVALID_TENSOR_CNT,
    SS_ERROR_UMD_TIMEOUT,
    SS_ERROR_UMD_NO_BATCH_QUEUE,
    SS_ERROR_UMD_MARK_SHARED_TENSOR,
    SS_ERROR_UMD_SET_SHARED_TENSOR,
    SS_ERROR_UMD_DMABUF_SHARED_IO,
    SS_ERROR_UMD_END,
    SS_ERROR_MAX,
} SessionResultStatus;

#endif
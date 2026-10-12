//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
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
 * @Last Modified time: 2025-10-17 17:43:59
 */

#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

#include "any.h"
#include "fp16.h"
#include "logger.h"
#include "memory_object.h"
#include "tensor_shape.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace aipubt
{
class Tensor;
class Node;
class Graph;
class Quantization;
class ComputeTensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorRef = std::weak_ptr<Tensor>;
using NodePtr = std::shared_ptr<Node>;
using NodeRef = std::weak_ptr<Node>;

/**
 * @brief a class for named Tensor descriptor, the tensor may have data, may not if it just a descriptor.
 *
 */
class Tensor : public std::enable_shared_from_this<Tensor>
{
public:
    /**
     * @brief the tensor data type
     *
     */
    enum Dtype : int
    {
        VOID = 0x0,
        BOOL = 0x1,
        UINT8 = 0x2,
        INT8 = 0x3,
        UINT16 = 0x4,
        INT16 = 0x5,
        UINT32 = 0x6,
        INT32 = 0x7,
        UINT64 = 0x8,
        INT64 = 0x9,
        FP16 = 0xa,
        FP32 = 0xb,
        FP64 = 0xc,
        BFP16 = 0xd,
        FP8E4M3FN = 0xe,
        FP8E5M2 = 0xf,

        // byte-aligned int4
        ALIGNED_UINT4 = 0x12, // UINT8+0x10
        ALIGNED_INT4 = 0x13,
        // byte-aligned int12
        ALIGNED_UINT12 = 0x14,
        ALIGNED_INT12 = 0x15,
        FP4E2M1FN = 0x16,

        //
        COMPACT_UINT4 = 0x20,
        COMPACT_INT4 = 0x21,
        COMPACT_UINT12 = 0x22,
        COMPACT_INT12 = 0x23,
        COMPACT_FP4E2M1FN = 0x24,
    };

    enum CompressAlgo
    {
        UNCOMPRESS = 0,
        BITMAP8 = 2,
        LEGACY8 = 2,

        BITMAP16 = 3,
        LEGACY16 = 3,

        LOWBIT8 = (1 << 2) + 2,
        LOWBIT16 = (1 << 2) + 3,

        X3_SUPPORTED = (2 << 2),
        I8ALLONE = (2 << 2) + 2,
        I8_333331 = (3 << 2) + 2,
        I4_772 = (4 << 2) + 1,
        I4_333331 = (5 << 2) + 1,
        I4ALLONE = (6 << 2) + 1,
    };

    struct CompressInfo
    {
        CompressAlgo algo = UNCOMPRESS;
        size_t ori_size = 0u;
        union
        {
            int8_t bits;
            uint32_t next_len;
            char lut[16];
        } extra = {0};
        int get_nbits()
        {
            if (algo == CompressAlgo::LOWBIT8 || algo == CompressAlgo::LOWBIT16)
            {
                return extra.bits;
            }
            return 0;
        }
        int get_head_len()
        {
            if (algo == CompressAlgo::BITMAP8 || algo == CompressAlgo::BITMAP16)
            {
                return extra.next_len;
            }
            return 0;
        }
        char *get_lut()
        {
            if (algo >= CompressAlgo::I8ALLONE)
            {
                return extra.lut;
            }
            return 0;
        }
        Dtype type;
        std::vector<int> shape;
    };
    struct CompressFrameInfo
    {
        /**
         * for bit map
         *  the layout is
         *  l*1024
         *  then repeat n*1024+m*tail until end
         *
         */
        uint64_t value;

        CompressFrameInfo()
        {
            encode(1, 0, 0, 0);
        }
        CompressFrameInfo(uint64_t v) : value(v)
        {
        }
        CompressFrameInfo(uint32_t n, uint32_t m, uint32_t tail, uint32_t l = 0)
        {
            encode(n, m, tail, l);
        }
        uint64_t encode(uint32_t n, uint32_t m, uint32_t tail, uint32_t l = 0)
        {
            // encode into 64 bit
            // l:24, n:24, m: 14 tail: 2;
            value = (((uint64_t)n & 0xffffff) << 16) | ((m & 0x3fff) << 2) | (tail & 0b11) | (((uint64_t)l & 0xffffff) << 40);
            return value;
        }
        operator uint64_t()
        {
            return value;
        }
    };
    static Tensor::Dtype string_to_dtype(const std::string &tstr);
    // get a int dtype by bits and sign, bits only support 8, 16, 32, 64, a void  type will return  if bits is invalid
    static Tensor::Dtype get_type(const int bits, const bool is_sign);
    static const std::string &dtype_to_string(Tensor::Dtype dt);

    static bool is_float(Tensor::Dtype dt)
    {
        return (dt >= Tensor::FP16 && dt <= Tensor::FP8E5M2) ||
               (dt == Tensor::FP4E2M1FN || dt == Tensor::COMPACT_FP4E2M1FN);
    }

    static bool is_compact(Tensor::Dtype dt)
    {
        return dt >= Tensor::COMPACT_UINT4 && dt <= Tensor::COMPACT_FP4E2M1FN;
    }

    static size_t dtype_bits(Tensor::Dtype dt)
    {
        if (dt <= 1)
        {
            return 8;
        }
        if (is_float(dt))
        {
            switch (dt)
            {
            case BFP16:
                return 16;
            case FP8E4M3FN:
            case FP8E5M2:
            case FP4E2M1FN:
                return 8;
            case COMPACT_FP4E2M1FN:
                return 4;
            default:
                return (1 << (dt - Tensor::FP16)) * 16;
            }
        }
        if (is_compact(dt))
        {
            return (1 << ((dt & 0xe) >> 1)) * 8 - 4;
        }
        return 8 << (((dt & 0xe) >> 1) - 1);
    }

    static size_t dtype_actual_bits(Tensor::Dtype dt)
    {
        if (dt <= 1)
        {
            return (dt & 0x1) * 1;
        }
        if (is_aligned(dt) || is_compact(dt))
        {
            int32_t offset = 0;
            if (dt <= Tensor::FP4E2M1FN)
                offset = 0xe;
            return (1 << ((((dt + offset) & 0x23) & 0xe) >> 1)) * 8 - 4;
        }
        if (is_float(dt))
        {
            switch (dt)
            {
            case BFP16:
                return 16;
            case FP8E4M3FN:
            case FP8E5M2:
                return 8;
            default:
                return (1 << (dt - Tensor::FP16)) * 16;
            }
        }
        return 8 << (((dt & 0xe) >> 1) - 1);
    }

    static size_t dtype_size(Tensor::Dtype dt)
    {
        return dtype_bits(dt) / 8.0f + 0.5f;
    }

    static Tensor::Dtype dtype_to_signed(Tensor::Dtype dt);
    static Tensor::Dtype dtype_to_unsigned(Tensor::Dtype dt);
    static bool is_signed(Tensor::Dtype dt);

    static TensorPtr workspace(size_t size);
    /**
     * @brief check is the dtype is interger type, not including align and compact int type
     *
     * @param dt
     * @return true
     * @return false
     */
    static bool is_int(Tensor::Dtype dt);
    static bool is_aligned(Tensor::Dtype dt);
    static bool fill_tensors_data(std::vector<TensorPtr> &tensors, std::vector<std::string> &files);

    friend std::ostream &operator<<(std::ostream &os, Tensor::Dtype dtype);
    friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
    friend std::ostream &operator<<(std::ostream &os, TensorPtr t);

    Tensor(const TensorShape &shape, Dtype type = UINT8);
    Tensor(void *data, size_t size);
    Tensor(const std::string &name, const TensorShape &shape, Dtype type = UINT8);
    Tensor(const std::string &name, void *data = 0, size_t size = 0);
    Tensor() = delete;
    /**
     * @brief Construct a new Tensor object
     *        construct a tensor by copy other, please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     *
     * @param t other Tensor
     */
    Tensor(const Tensor &t);

    /**
     * @brief Construct a new Tensor object
     *        construct a tensor by copy other, please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     *
     * @param t other Tensor
     */
    Tensor &operator=(const Tensor &t);
    virtual ~Tensor();
    // return a tensor by name if the tensor has be created, else return null
    // static TensorPtr get_tensor(const string &tensor_name);
    /**
     * @brief get the tensor's data if available else nullptr will be return,
     * the data's memory is owned by tensor, client only can read/write it, no create/delete operation
     *
     * @tparam T the data type
     * @return T* the data pointer
     */
    template <typename T>
    T *data()
    {
        return reinterpret_cast<T *>(m_data + m_data_offset);
    }

    int8_t *data()
    {
        return reinterpret_cast<int8_t *>(m_data + m_data_offset);
    }

    template <typename T>
    T *device_data()
    {
        return reinterpret_cast<T *>(m_device_data + m_data_offset);
    }

    int8_t *device_data()
    {
        return reinterpret_cast<int8_t *>(m_device_data + m_data_offset);
    }

    /**
     * @brief get/set the tensor shape
     *
     * @return TensorShape& the tensor shape
     */
    virtual TensorShape &shape()
    {
        return m_shape;
    }

    /**
     * @brief get the tensor element number
     *
     * @return size_t the tensor size
     */
    virtual size_t size() const
    {
        // if (m_offset->size > 0)
        // {
        //     return m_offset->size;
        // }
        return m_shape.size();
    }

    /**
     * @brief get the memory size needed by the tensor where the strides of the tensor is not considered.
     *
     * @return size_t memory size
     */
    virtual size_t mem_size() const;

    /**
     * @brief get the buffer size occupyied by the tensor, which will count the strides of the tensor
     *
     * @return size_t
     */
    virtual size_t buffer_size() const;

    /**
     * @brief get the memory object of this tensor
     *
     * @return MemoryObjectPtr memory object
     */
    virtual MemoryObjectPtr memory_offset()
    {
        return m_offset;
    }

    /**
     * @brief get the memory offset of this tensor
     *
     * @return uint32_t offset
     */
    virtual uint64_t offset()
    {
        return m_offset->offset();
    }

    /**
     * @brief Set the data to the tensor, then the data's memory ownership will transer to tensor
     *
     * @param d the data
     * @param size size of data
     */
    virtual void set_data(void *d, size_t size);
    virtual void set_data(void *d, size_t size, bool is_owner);
    /**
     * @brief Set the device data to the tensor, then the  data's memory ownership is indicated by owner parameter
     *
     * @param d the data
     * @param size size of data
     * @param owner the owner of device_data: if true, owned by this Tensor, otherwise, owned by others
     */
    virtual void set_device_data(void *d, size_t size, const bool owner = true);
    bool is_owns_host_data() const
    {
        return *m_owner_host;
    }
    bool is_owns_device_data() const
    {
        return *m_owner_device;
    }
    // free old data and share a new data of t
    /**
     * @brief share data from other tensor
     *
     * @param t other tensor
     */
    virtual void share_data_from(Tensor &t);
    void share_data_from(TensorPtr t)
    {
        share_data_from(*t);
    }

    /**
     * @brief Set the data type
     *
     * @param t data type
     */
    virtual void set_type(Dtype t)
    {
        m_type = t;
    }

    /**
     * @brief get the data type
     *
     * @return Dtype data type
     */
    virtual Dtype type() const
    {
        return m_type;
    }

    /**
     * @brief get data type's string
     *
     * @return const std::string&  data type string
     */
    virtual const std::string &type_s() const;

    /**
     * @brief get tensor name
     *
     * @return const std::string& tensor name
     */
    virtual const std::string &name() const
    {
        return m_name;
    }
    /**
     * @brief Set the tensor name
     *
     * @param name tensor name
     */
    virtual void set_name(const std::string &name = "");

    /**
     * @brief get the quantization parameters of this tensor
     *
     * @return const Quantization the quant param
     */
    virtual const Quantization quantization() const;
    /**
     * @brief get/set the quantization parameters of this tensor
     *
     * @return Quantization& the quant param
     */
    virtual Quantization &quantization();

    // Compact the byte-aligned data according to its actual bits
    /**
     * @brief compact the tensor data if the data type is aligned data type: aligned_int4/12
     *
     */
    virtual void compact();

    /**
     * @brief convert compact data into aligned data
     *
     */
    virtual void decompact();

    /**
     * @brief get the compat memory size if the data type is aligned data type: aligned_int4/12
     *
     * @return size_t compat memory size
     */
    virtual size_t compact_mem_size() const;

    /**
     * @brief check if the tensor is compat
     *
     * @return true
     * @return false
     */
    virtual bool is_compact() const;

    /**
     * @brief get the node which own the tensor, null will be return if no node present
     *
     * @return NodePtr
     */
    NodePtr node() const;

    /**
     * @brief return the consumers of this tensor
     *
     * @return std::vector<NodePtr>
     */
    std::vector<NodePtr> consumers() const;

    /**
     * @brief set the tensor's attribute to be an input or output tensor
     *
     * @param bool true for io tensor, false for other tensor
     */
    virtual void set_io_tensor(bool iot);

    /**
     * @brief check the tensor is an input or output tensor
     *
     * @return bool
     */
    virtual bool is_io_tensor() const;

    /**
     * @brief set the tensor's attribute to be a fake tensor
     *
     * @param bool true for fake tensor, false for other tensor
     */
    virtual void set_fake_tensor(bool fake);

    /**
     * @brief check the tensor is a fake tensor
     *
     * @return bool
     */
    virtual bool is_fake_tensor() const;

    /**
     * @brief set the tensor's attribute to be a prior in gm tensor
     *
     * @param bool true for io tensor, false for other tensor
     */
    virtual void set_prior_tensor(bool iot);

    /**
     * @brief check the tensor is a prior in gm tensor
     *
     * @return bool
     */
    virtual bool is_prior_tensor() const;

    /**
     * @brief set the tensor's attribute to be a const tensor
     *
     * @param bool true for const tensor, false for other tensor
     */
    virtual void set_const_tensor(bool iot);

    /**
     * @brief check the tensor is a const tensor
     *
     * @return bool
     */
    virtual bool is_const_tensor() const;

    /**
     * @brief get the attributes map of this tensor
     *
     * @return std::map<std::string, Any>& attributes map
     */
    // DEPRECATED("for safe using attributes and avoiding memory leak, please use set_attr and get_attr")
    virtual std::map<std::string, Any> &attrs()
    {
        return m_attrs;
    }

    virtual const std::map<std::string, Any> &attrs() const
    {
        return m_attrs;
    }
    virtual void set_attr(const std::string &key, const Any &value)
    {
        m_attrs[key] = value;
    }
    virtual void erase_attr(const std::string &key)
    {
        m_attrs.erase(key);
    }
    virtual Any get_attr(const std::string &key) const
    {
        return m_attrs.at(key);
    }
    template <typename T>
    inline T get_attr(const std::string &key) const
    {
        if (attrs().count(key))
        {
            return cast_any<T>(attrs().at(key));
        }
        else
        {
            return T();
        }
    }
    virtual bool operator==(const Tensor &other) const;
    virtual bool operator!=(const Tensor &other) const;

    template <typename T>
    T max()
    {
        if (m_data)
        {
            auto datas = reinterpret_cast<T *>(m_data);
            auto size = this->mem_size() / sizeof(T);
            T max_val = *std::max_element(datas, datas + size);
            return max_val;
        }
        else
        {
            ERROR() << "Tensor Doesn't have initialzed avaliable data.";
            return T();
        }
    }

    template <typename T>
    T min()
    {
        if (m_data)
        {
            auto datas = reinterpret_cast<T *>(m_data);
            auto size = this->mem_size() / sizeof(T);
            T min_val = *std::min_element(datas, datas + size);
            return min_val;
        }
        else
        {
            ERROR() << "Tensor Doesn't have initialzed avaliable data.";
            return T();
        }
    }
    void make_memory_continuous();
    void monopolize_data(); // make no tensor share this data

    TensorPtr compress(float rate = 0.98, const std::vector<int> algos = std::vector<int>());
    const CompressInfo get_compress_info();
    TensorPtr decompress();
    int get_ref_count();
    // transfer ownership, if the data is not owner by the Tensor, nothing happen, return false
    // if the data is owned by the Tensor, it will transfer to others, and the Tensor will not free the data.
    bool transfer_ownership(bool is_host = true);

protected:
    void _set_host_device_data(void *d, const size_t &size, const bool owner, const bool is_host);
    void _clean_data();

    template <Tensor::Dtype dtype>
    void compact_impl();
    template <Tensor::Dtype dtype>
    void decompact_impl();
    void add_consumer(NodePtr n);
    void remove_consumer(NodePtr n);

    std::string m_name;
    char *m_data = nullptr;
    char *m_device_data = nullptr;
    size_t m_data_offset = 0;
    // check m_data ref
    std::atomic<int> *ref_count = nullptr;
    std::atomic<bool> *m_owner_host = nullptr;
    std::atomic<bool> *m_owner_device = nullptr;
    TensorShape m_shape;
    Dtype m_type = UINT8;
    MemoryObjectPtr m_offset = MemoryObjectPtr(new MemoryObject(0));
    std::map<std::string, Any> m_attrs;
    NodeRef m_node;
    std::vector<NodeRef> m_consumers;
    //0 : io
    //1 : fake
    //2 : prior
    //3 : ConstTensor
    uint64_t m_flag = 0;
    friend Node;
    friend ComputeTensor;
};

const std::map<std::string, Tensor::Dtype> &get_dtype_dict();
TensorPtr shrink_tensor_data(const TensorPtr t, int axis = -1, bool keep_dim = false);

#define _TENSOR_TYPE_MAP(Op)          \
    Op(float, Tensor::FP32);          \
    Op(double, Tensor::FP64);         \
    Op(half, Tensor::FP16);           \
    Op(bfp16, Tensor::BFP16);         \
    Op(fp8e4m3fn, Tensor::FP8E4M3FN); \
    Op(fp8e5m2, Tensor::FP8E5M2);     \
    Op(fp4e2m1fn, Tensor::FP4E2M1FN); \
    Op(int64_t, Tensor::INT64);       \
    Op(uint64_t, Tensor::UINT64);     \
    Op(int32_t, Tensor::INT32);       \
    Op(uint32_t, Tensor::UINT32);     \
    Op(int16_t, Tensor::INT16);       \
    Op(uint16_t, Tensor::UINT16);     \
    Op(int8_t, Tensor::INT8);         \
    Op(uint8_t, Tensor::UINT8);

#define _TENSOR_TYPE_MAP_ALIGN(Op)      \
    Op(int8_t, Tensor::ALIGNED_INT4);   \
    Op(uint8_t, Tensor::ALIGNED_UINT4); \
    Op(int16_t, Tensor::ALIGNED_INT12); \
    Op(uint16_t, Tensor::ALIGNED_UINT12);

template <Tensor::Dtype>
struct tensor_data_type
{
    typedef void type;
};

#define TENSOR_DATA_TYPE(t, T) \
    template <>                \
    struct tensor_data_type<T> \
    {                          \
        typedef t type;        \
    };

_TENSOR_TYPE_MAP(TENSOR_DATA_TYPE);
_TENSOR_TYPE_MAP_ALIGN(TENSOR_DATA_TYPE);

template <typename>
struct tensor_type
{
};

#define DECLARE_TENSOR_TYPE(t, T)              \
    template <>                                \
    struct tensor_type<t>                      \
    {                                          \
        static constexpr Tensor::Dtype value() \
        {                                      \
            return T;                          \
        }                                      \
    };
_TENSOR_TYPE_MAP(DECLARE_TENSOR_TYPE)

template <typename>
struct is_supported_tensor_type
{
    static constexpr bool value = false;
};

#define DECLARE_SUPPORTED_TENSOR_TYPE(t, T) \
    template <>                             \
    struct is_supported_tensor_type<t>      \
    {                                       \
        static constexpr bool value = true; \
    };
_TENSOR_TYPE_MAP(DECLARE_SUPPORTED_TENSOR_TYPE)

} // namespace aipubt
#endif //! CORE_TENSOR_
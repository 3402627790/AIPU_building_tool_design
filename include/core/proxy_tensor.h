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
 * @Last Modified time: 2024-08-23 10:02:57
 */
#ifndef CORE_PROXY_TENSOR_H
#define CORE_PROXY_TENSOR_H

#include "tensor.h"

namespace aipubt
{

class ProxyTensor;
using ProxyTensorPtr = std::shared_ptr<ProxyTensor>;
/**
 * @brief a class for Proxy Tensor, which is a tensor proxy, 
 * all atttributes and actions are same as its proxied tensor, 
 * except the node(owner) and consumer will be different.
 *
 */
class ProxyTensor : public Tensor
{
public:
    ProxyTensor() = delete;
    /**
     * @brief Construct a new Tensor object
     *        construct a tensor by copy other, please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     * 
     * @param t other Tensor
     */
    ProxyTensor(const TensorPtr t);

    /**
     * @brief Construct a new Tensor object
     *        construct a tensor by copy other, please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     * 
     * @param t other Tensor
     */
    ProxyTensor &operator=(const TensorPtr t);
    ~ProxyTensor();

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
        return m_tensor->data<T>();
    }

    /**
     * @brief get/set the tensor shape
     *
     * @return TensorShape& the tensor shape
     */
    TensorShape &shape() override;

    /**
     * @brief get the tensor size, if has data, return the data size, else return the shape size.
     *
     * @return size_t the tensor size
     */
    size_t size() const override;

    /**
     * @brief get the memory size needed by the tensor
     *
     * @return size_t memory size
     */
    size_t mem_size() const override;

    size_t buffer_size() const override;

    /**
     * @brief get the memory object of this tensor
     *
     * @return MemoryObjectPtr memory object
     */
    MemoryObjectPtr memory_offset() override;

    /**
     * @brief get the memory offset of this tensor
     *
     * @return uint32_t offset
     */
    uint32_t offset() override;

    /**
     * @brief Set the data to the tensor, then the data's memory ownership will transer to tensor
     *
     * @param d the data
     * @param size size of data
     */
    void set_data(void *d, uint32_t size) override;
    // free old data and share a new data of t
    /**
     * @brief share data from other tensor
     *
     * @param t other tensor
     */
    void share_data_from(Tensor &t) override;

    /**
     * @brief Set the data type
     *
     * @param t data type
     */
    void set_type(Dtype t) override;

    /**
     * @brief get the data type
     *
     * @return Dtype data type
     */
    Dtype type() const override;

    /**
     * @brief get data type's string
     *
     * @return const std::string&  data type string
     */
    const std::string &type_s() const override;

    /**
     * @brief get tensor name
     *
     * @return const std::string& tensor name
     */
    const std::string &name() const override;

    /**
     * @brief Set the tensor name
     *
     * @param name tensor name
     */
    void set_name(const std::string &name = "") override;

    /**
     * @brief get the quantization parameters of this tensor
     *
     * @return const Quantization the quant param
     */
    const Quantization quantization() const override;
    /**
     * @brief get/set the quantization parameters of this tensor
     *
     * @return Quantization& the quant param
     */
    Quantization &quantization() override;

    // Compact the byte-aligned data according to its actual bits
    /**
     * @brief compat the tensor data if the data type is aligned data type: aligned_int4/12
     *
     */
    void compact() override;

    /**
     * @brief get the compat memory size if the data type is aligned data type: aligned_int4/12
     *
     * @return size_t compat memory size
     */
    size_t compact_mem_size() const override;

    /**
     * @brief check if the tensor is compat
     *
     * @return true
     * @return false
     */
    bool is_compact() const override;

    /**
     * @brief set the tensor's attribute to be an input or output tensor
     * 
     * @param bool true for io tensor, false for other tensor
     */
    void set_io_tensor(bool iot) override;

    /**
     * @brief check the tensor is an input or output tensor
     * 
     * @return bool 
     */
    bool is_io_tensor() const override;

    /**
     * @brief get the attributes map of this tensor
     *
     * @return std::map<std::string, Any>& attributes map
     */
    std::map<std::string, Any> &attrs() override;
    const std::map<std::string, Any> &attrs() const override;

    void set_attr(const std::string &key, const Any &value) override;
    Any get_attr(const std::string &key) const override;
    void erase_attr(const std::string &key) override;

    bool operator==(const Tensor &other) const override;
    bool operator!=(const Tensor &other) const override;

    TensorPtr proxied_tensor(bool recurrently = false) const
    {
        auto pt = std::dynamic_pointer_cast<ProxyTensor>(m_tensor);
        if (recurrently && pt)
        {
            return pt->proxied_tensor();
        }
        return m_tensor;
    }

protected:
    TensorPtr m_tensor;
    friend Node;

private:
    //hide some  functions
    using Tensor::operator=;
};

} // namespace aipubt
#endif //!CORE_PROXY_TENSOR_H

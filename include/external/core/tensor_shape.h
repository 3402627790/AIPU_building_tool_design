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
 * @Date: 2020-02-09 12:16:03
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-11-08 11:13:11
 */
#ifndef CORE_TENSOR_SHAPE_H
#define CORE_TENSOR_SHAPE_H

#include <assert.h>
#include <functional>
#include <initializer_list>
#include <string>
#include <tongue/symbolic.h>
#include <vector>
namespace aipubt
{
/**
 * @brief Tensor layout, Flat is no layout.
 * NCHWCx is NCHW and Cx aligned. for example, NCHWC4 layout has a shape [n,c0,h,w,4], the total channel is c0*4
 * NCHWCxTy is NCHW and Cx aligned and the tail is y align. for example, NCHWC32T8 layout has a shape [n,c0,h,w,32]+[n,h,w,y*k], the total channel is c0*32+y*k where k is interger
 * the layout with sparse suffix is storage in sparse format.
 *
 */
enum DataLayout : int
{
    // layout 0-15 for normal
    // 16-32 for tail
    // 32-64 for sparse
    DataLayout_UnSupport = -1,
    DataLayout_Unkown = 0,
    DataLayout_Flat = 0,
    DataLayout_NCHW = 1,
    DataLayout_NHWC = 2,
    DataLayout_NCHWC4 = 3,
    DataLayout_NCHWC8 = 4,
    DataLayout_NCHWC16 = 5,
    DataLayout_NCHWC32 = 6,

    DataLayout_NDHWC = 8,
    DataLayout_NDCHWC16 = 9,
    DataLayout_NDCHWC32 = 10,

    DataLayout_NCDHWC16 = 11,
    DataLayout_NCDHWC32 = 12,

    // for with tail layout, the shape is stored in NHWC,
    //  but when you get C, it would return the C just same as NCHWCxTx'c, you should get out_c
    //  for example  a NCHWC32T8,  [1,H,W,48], c=48/32, c_tail=48%32, c_total=48, c_prime=32
    DataLayout_NCHWC16T8 = DataLayout_NCHWC16 + 16,
    DataLayout_NCHWC32T8 = DataLayout_NCHWC32 + 16,
    DataLayout_NCHWC16T1 = DataLayout_NCHWC16T8 + 2,
    DataLayout_NCHWC32T1 = DataLayout_NCHWC32T8 + 2,
    DataLayout_NDCHWC16T8 = DataLayout_NDCHWC16 + 16,
    DataLayout_NDCHWC32T8 = DataLayout_NDCHWC32 + 16,
    DataLayout_NDCHWC16T1 = DataLayout_NDCHWC16T8 + 2,
    DataLayout_NDCHWC32T1 = DataLayout_NDCHWC32T8 + 2,

    DataLayout_NHWCSparse = DataLayout_NHWC + 32,
    DataLayout_NCHWC16Sparse = DataLayout_NCHWC16 + 32,
    DataLayout_NCHWC32Sparse = DataLayout_NCHWC32 + 32,
    DataLayout_NCHWC16T8Sparse = DataLayout_NCHWC16T8 + 48,
    DataLayout_NCHWC32T8Sparse = DataLayout_NCHWC32T8 + 48,
    DataLayout_NCHWC16T1Sparse = DataLayout_NCHWC16T8Sparse + 64,
    DataLayout_NCHWC32T1Sparse = DataLayout_NCHWC32T8Sparse + 64,

    DataLayout_MaxCount,
};

typedef std::vector<DataLayout> LayoutList;
typedef std::vector<LayoutList> LayoutCombinations;
std::ostream &operator<<(std::ostream &os, const DataLayout layout);
DataLayout string_to_datalayout(const std::string &str);
std::string datalayout_to_string(DataLayout layout);
class TensorShape;

/**
 * @brief union class for shape dimension index: n,c,h,w
 *
 */
struct ShapeDim
{
    int n : 4;
    int h : 4;
    int w : 4;
    int c : 4;
    int d : 4;
};

/**
 * @brief a class for represent the symbolic shape
 *
 */
class SymbolShape
{
public:
    SymbolShape() {}
    SymbolShape(const std::vector<tongue::Expr> &shape);
    SymbolShape(const std::vector<tongue::Expr> &shape, DataLayout layout, ShapeDim idx);
    SymbolShape(const SymbolShape &shape);

    /**
     * @brief get the index axis dimension
     *
     * @param index the axis index
     * @return tongue::Expr the dimension of axis
     */
    tongue::Expr operator[](int index) const;
    std::vector<tongue::Expr>::iterator begin();
    std::vector<tongue::Expr>::const_iterator begin() const;
    std::vector<tongue::Expr>::iterator end();
    std::vector<tongue::Expr>::const_iterator end() const;

    operator std::vector<tongue::Expr>() const;
    bool operator==(const SymbolShape &other) const;
    bool operator!=(const SymbolShape &other) const;
    /**
     * @brief get the h dimension, if the layout is has no H, error raised.
     *
     * @return tongue::Expr the h dim
     */
    tongue::Expr h() const;

    /**
     * @brief get the w dimension, if the layout is has no W, error raised.
     *
     * @return tongue::Expr the w dim
     */
    tongue::Expr w() const;

    /**
     * @brief get the c dimension, if the layout is has no C, error raised.
     *
     * @return tongue::Expr the c dim, for NCHWCx, it return c0
     */
    tongue::Expr c() const;

    /**
     * @brief get/set the n dimension, if the layout is has no N, error raised.
     *
     * @return tongue::Expr the n dim
     */
    tongue::Expr n() const;

    /**
     * @brief get the c' dimension, if the layout is has no C', error raised.
     *
     * @return tongue::Expr the c dim, just for NCHWCx, it return c0
     */
    tongue::Expr c_prime() const;

    /**
     * @brief get the c tail dimension, if the layout is has no C tail, error raised.
     *
     * @return tongue::Expr the c tail dim, just for NCHWCx, it return Cx
     */
    tongue::Expr c_tail() const;

    /**
     * @brief get the c total dimension, if the layout is has no C tail, error raised.
     *
     * @return tongue::Expr the c tail dim, for NCHWCx, it return Cx*C0, other return C
     */
    tongue::Expr c_total() const;

    /**
     * @brief get the total size of the shape
     *
     * @return tongue::Expr
     */
    tongue::Expr size() const;

    /**
     * @brief get dimension number of the shape
     *
     * @return int
     */
    int dim() const
    {
        return m_shape.size();
    }

    /**
     * @brief get a string of the shape
     *
     * @return const std::string
     */
    const std::string to_string() const;

    /**
     * @brief check if the shape is constant
     *
     * @return bool
     */
    bool is_constant() const;

    /**
     * @brief get the layout of shape
     *
     * @return DataLayout the layout
     */
    DataLayout layout() const
    {
        return m_layout;
    }

    /**
     * @brief get the shape indexing
     *
     * @return ShapeDim
     */
    ShapeDim indexing() const
    {
        return m_idx;
    }

private:
    ShapeDim m_idx = {0, 1, 2, 3, -1}; // default dim is -1, -1 means that has no D axis
    DataLayout m_layout = DataLayout_Flat;
    std::vector<tongue::Expr> m_shape;
    friend class TensorShape;
};

/**
 * @brief a class for represent the tensor shape
 *
 */
class TensorShape
{
public:
    // TensorShape(int d0 = 1, DataLayout dl = DataLayout_Unkown);
    TensorShape(std::initializer_list<int> l, DataLayout dl = DataLayout_Unkown);
    TensorShape(const std::vector<int> &shape, DataLayout dl = DataLayout_Unkown);
    TensorShape(std::vector<int> &shape, DataLayout dl = DataLayout_Unkown) : TensorShape((const std::vector<int> &)shape, dl) {}
    TensorShape() {}
    TensorShape(int di)
    {
        push_back(di);
    }
    template <typename... IList>
    TensorShape(int di, IList... new_shape)
    {
        push_back(di, new_shape...);
    }
    TensorShape(const TensorShape &ts);
    TensorShape(const std::string &str);

    operator std::vector<int>() const;
    operator std::vector<uint32_t>() const;
    /**
     * @brief get the index axis dimension
     *
     * @param index the axis index
     * @return int& the dimension of axis
     */
    int &operator[](int index);
    /**
     * @brief get the index axis dimension
     *
     * @param index the axis index
     * @return int& the dimension of axis
     */
    int operator[](int index) const;
    TensorShape &operator=(const TensorShape &other);

    bool operator==(const TensorShape &other) const;
    bool operator>(const TensorShape &other) const;
    bool operator>=(const TensorShape &other) const
    {
        return (*this) == other || (*this) > other;
    }
    bool operator<(const TensorShape &other) const;
    bool operator<=(const TensorShape &other) const
    {
        return (*this) == other || (*this) < other;
    }
    bool operator!=(const TensorShape &other) const;
    std::vector<int>::iterator begin();
    std::vector<int>::const_iterator begin() const;
    std::vector<int>::iterator end();
    std::vector<int>::const_iterator end() const;

    /**
     * @brief get/set the h dimension, if the layout is has no H, error raised.
     *
     * @return int& the h dim
     */
    int &h();

    /**
     * @brief get/set the w dimension, if the layout is has no W, error raised.
     *
     * @return int& the w dim
     */
    int &w();

    /**
     * @brief get/set the c dimension, if the layout is has no C, error raised.
     *
     * @return int& the c dim, for NCHWCx, it return c0
     */
    int &c();

    /**
     * @brief get/set the n dimension, if the layout is has no N, error raised.
     *
     * @return int& the n dim
     */
    int &n();

    /**
     * @brief get the h dimension, if the layout is has no H, error raised.
     *
     * @return int& the h dim
     */
    int h() const;

    /**
     * @brief get the w dimension, if the layout is has no W, error raised.
     *
     * @return int& the w dim
     */
    int w() const;

    /**
     * @brief get the c dimension, if the layout is has no C, error raised.
     *
     * @return int& the c dim, for NCHWCx, it return c0
     */
    int c() const;

    /**
     * @brief get/set the n dimension, if the layout is has no N, error raised.
     *
     * @return int& the n dim
     */
    int n() const;

    /**
     * @brief get the c' dimension, if the layout is has no C', error raised.
     *
     * @return int& the c dim, just for NCHWCx, it return c0
     */
    int c_prime() const;

    /**
     * @brief get the c tail dimension, if the layout is has no C tail, error raised.
     *
     * @return int& the c tail dim, just for NCHWCx, it return Cx
     */
    int c_tail() const;

    /**
     * @brief get the c total dimension, if the layout is has no C tail, error raised.
     *
     * @return int& the c tail dim, for NCHWCx, it return Cx*C0, other return C
     */
    int c_total() const;

    /**
     * @brief get a string of the shape
     *
     * @return const std::string
     */
    const std::string to_string() const;

    /**
     * @brief get the total size of the shape
     *
     * @return size_t
     */
    int size() const;

    /**
     * @brief get dimension number of the shape
     *
     * @return int
     */
    int dim() const;
    // just set the layout

    /**
     * @brief Set the layout object, internal index would NOT update, not recommend to use it
     *
     * @param lo target layout
     */
    void set_layout(DataLayout lo);

    // change the layout from one to other, the shape also change
    /**
     * @brief change the layout to target layout, if failed, will change layout to Flat. For example, [1,2] cannot change to NHWC.
     *
     * @param layout target layout
     *
     * @return true if succ
     * @return false if failed
     */
    bool change_layout(DataLayout layout);

    /**
     * @brief get the layout of shape
     *
     * @return DataLayout the layout
     */
    DataLayout layout() const;

    /**
     * @brief reshape the shape to new shape
     *
     * @tparam IList
     * @param di dimension 0
     * @param new_shape dimension i...
     * @return true if succ
     * @return false if failed
     */
    template <typename... IList>
    bool reshape(int di, IList... new_shape)
    {
        TensorShape ts(di, new_shape...);
        ts.m_symbol.m_layout = m_symbol.m_layout;
        auto tsize = ts.size();
        if (tsize < 0)
        {
            int idx = -1;
            for (int i = 0; i < ts.dim(); ++i)
            {
                if (ts[i] == -1)
                {
                    if (idx == -1)
                    {
                        idx = i;
                    }
                    else
                    {
                        idx = -2;
                        break;
                    }
                }
            }
            if (idx >= 0)
            {
                ts[idx] = size() / (-tsize);
            }
        }
        if (ts.size() == size())
        {
            (*this) = ts;
            check_layout(true);
            return true;
        }

        return false;
    }
    /**
     * @brief reshape the shape to new shape
     *
     * @param l new shape
     * @return true if succ
     * @return false if failed
     */
    bool reshape(std::initializer_list<int> l);

    /**
     * @brief reshape the shape to new shape
     *
     * @param l new shape
     * @return true if succ
     * @return false if failed
     */
    bool reshape(std::vector<int> l);

    /**
     * @brief get the shape indexing
     *
     * @return ShapeDim
     */
    ShapeDim indexing() const
    {
        return m_symbol.m_idx;
    }

    /**
     * @brief get the strides
     *
     * @return std::vector<uint32_t>
     */
    std::vector<uint32_t> strides();

    /**
     * @brief reset the strides based on current shape
     *
     * @return std::vector<uint32_t>
     */
    std::vector<uint32_t> reset_strides();

    /**
     * @brief set the strides
     *
     * @return std::vector<uint32_t>
     */
    void set_strides(const std::vector<uint32_t> &strides);

    /**
     * @brief check the strides is legal
     *
     * @return std::vector<uint32_t>
     */
    bool check_strides(const std::vector<uint32_t> &strides);

    /**
     * @brief check if the shape stride is continuos
     *
     * @return true
     * @return false
     */
    bool is_continuous() const;

    /**
     * @brief check the shape is broadcastable with other, the layout will be ignored in broadcasting.
     *          the broadcasting rule is same as numpy's.
     *
     * @param t another shape
     * @return true
     * @return false
     */
    bool broadcastable(const TensorShape &t) const;

    /**
     * @brief calculate the broadcasting shape with another shape, the layout will be ignored in broadcasting
     *
     * @param t another shape
     * @return TensorShape the result, if not broadcatable, an empty shape will be return
     */
    TensorShape broadcasting(const TensorShape &t) const;

    /**
     * @brief get the symbolic shape corresponding to this tensor shape
     *
     * @return SymbolShape
     */
    const SymbolShape &symbol() const
    {
        return m_symbol;
    }
    /**
     * @brief get the symbolic shape corresponding to this tensor shape
     *
     * @return SymbolShape
     */
    SymbolShape &symbol()
    {
        return m_symbol;
    }

private:
    bool check_layout(bool is_correct = false);
    void push_back(int d);
    template <typename... IList>
    void push_back(int di, IList... new_shape)
    {
        push_back(di);
        push_back(new_shape...);
    }

    std::vector<int> m_shape;
    std::vector<uint32_t> m_strides;
    bool m_shape_change = false;
    SymbolShape m_symbol;
};

std::vector<uint32_t> offset_to_index(size_t offset, const std::vector<uint32_t> &stride);
uint32_t index_to_offset(const std::vector<uint32_t> &index, const std::vector<uint32_t> &stride);

std::function<int(int)> get_datalayout_axis_map(DataLayout from, DataLayout to);

std::ostream &operator<<(std::ostream &os, const SymbolShape &shape);
std::ostream &operator<<(std::ostream &os, const TensorShape &shape);
typedef std::vector<TensorShape> ShapeList;

TensorShape get_base_shape(const TensorShape &shape);
} // namespace aipubt

#endif //! CORE_TENSOR_SHAPE_H

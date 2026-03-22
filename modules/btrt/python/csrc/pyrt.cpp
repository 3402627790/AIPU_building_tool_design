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
 * @Date: 2021-06-21 17:42:42
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 14:56:40
 */
#include "pyrt.h"
#include "btrt_api.h"
#include "sim_device.h" //TODO support more devices
#include <tuple>
using namespace aipurt;

PYBIND11_MODULE(_btrt, m)
{
    py::class_<SimDevice>(m, "SimDevice", py::dynamic_attr())
        .def(py::init())
        .def("load", &SimDevice::load)
        .def("allocate", [] (SimDevice &self, size_t size, size_t algin) {
            deviceptr addr=0;
            if(self.allocate(&addr, size, algin) != 0)
            {
                throw std::runtime_error("Failed to allocate device memory");
            }
            return addr; })
        .def("free", [] (SimDevice &self, deviceptr ptr) {
            if(self.deallocate(ptr) != 0)
            {
                throw std::runtime_error("Failed to free device memory");
            } })
        .def("copyH2D", [](SimDevice &self, deviceptr device, py::object &host, size_t size) {
            auto arr = pybind11::array::ensure(host);
            if (!arr)
            {
                throw std::runtime_error("expect a numpy array as host ptr");
            }
            auto buf_info = arr.request();
            uint64_t mem_size = buf_info.size * buf_info.itemsize;
            if(mem_size < size)
            {
                throw std::runtime_error("numpy array is smaller than the required size");
            }
            self.copyH2D(device, buf_info.ptr, size);
            })
        .def("copyD2H", [](SimDevice &self, py::object &host, deviceptr device, size_t size) {
            auto arr = pybind11::array::ensure(host);
            if (!arr)
            {
                throw std::runtime_error("expect a numpy array as host ptr");
            }
            auto buf_info = arr.request();
            uint64_t mem_size = buf_info.size * buf_info.itemsize;
            if(mem_size < size)
            {
                throw std::runtime_error("numpy array is smaller than the required size");
            }
            self.copyD2H(buf_info.ptr, device, size);
            })
        .def("copyD2D", [](SimDevice &self, deviceptr device, py::object &host, size_t size) {
                uint64_t src = py::cast<uint64_t>(host);
                self.copyD2D(device, src, size);
            })

        .def_property_readonly("symtab", [](SimDevice &self) { return self.kernel_names(); })
        .def("launch", [] (SimDevice &self, const std::string &func_name, const std::vector<uint32_t> &args, std::tuple<int, int, int> grid, std::tuple<int, int, int> block) {
            dim3_t g = {(uint16_t)std::get<0>(grid), (uint16_t)std::get<1>(grid), (uint16_t)std::get<2>(grid)};
            dim3_t b = {(uint16_t)std::get<0>(block), (uint16_t)std::get<1>(block), (uint16_t)std::get<2>(block)};
            self.launch(func_name.c_str(), g, b, args);
        }

        );
}

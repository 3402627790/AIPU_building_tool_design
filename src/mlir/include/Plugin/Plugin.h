//===------------------- Plugin.h - AIPU Pass Interface ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIPU external pass interface
//
//===----------------------------------------------------------------------===//
#include "mlir/Pass/Pass.h"
#include <memory>

#ifndef AIPUGC_PLUGIN_PLUGIN_H
#define AIPUGC_PLUGIN_PLUGIN_H

#if defined(_WIN32)
#define PLUGIN_EXPORT __declspec(dllexport)
#else
#define PLUGIN_EXPORT
#endif

enum PluginType {
  PluginTypePass = 0x0,
};
using PassFunc = std::function<std::unique_ptr<mlir::Pass>(void)>;
struct PluginContext {
  unsigned int plugin_type;
  std::string pass_name;
  PassFunc pass_func;
};

namespace mlir {
namespace aipugc {
void AipuRegisterPlugin(PluginContext *plugin);
}; // namespace aipugc
}; // namespace mlir

class PluginRegisterHelper {
public:
  PluginRegisterHelper(PluginContext *plugin) {
    mlir::aipugc::AipuRegisterPlugin(plugin);
  }
};
#define REGISTER_PASS_PLUGIN(pass)                                             \
  std::unique_ptr<mlir::Pass> create##pass(void) {                             \
    return std::make_unique<pass>();                                           \
  }                                                                            \
  PLUGIN_EXPORT PluginContext pluginContext##pass = {                          \
      PluginTypePass,                                                          \
      #pass,                                                                   \
      create##pass,                                                            \
  };                                                                           \
  static PluginRegisterHelper pluginRegister##pass(                            \
      (PluginContext *)&pluginContext##pass);

#endif // AIPUGC_PLUGIN_PLUGIN_H
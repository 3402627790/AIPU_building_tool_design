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
#include "Plugin/Plugin.h"
#include "mlir/Pass/PassManager.h"

#ifndef AIPUGC_PLUGIN_PLUGINLOADER_H
#define AIPUGC_PLUGIN_PLUGINLOADER_H

#include <map>
#include <set>
#include <string>
#include <vector>

namespace mlir {
namespace aipugc {

class PluginLoader {
public:
  static PluginLoader *getInstance();
  void setPluginPath(std::string path);
  std::vector<PassFunc> pluginPasses();
  bool verity();

private:
  PluginLoader();
  ~PluginLoader();
  void loadAllPlugin(const char *libPath);
  void loadPlugin(const std::string &filename);
  std::map<std::string, void *> handles;
  std::string pluginPath;
};

void AipuApplyPassPlugin(mlir::OpPassManager &optPM);

} // namespace aipugc
} // namespace mlir

#endif // AIPUGC_PLUGIN_PLUGINLOADER_H

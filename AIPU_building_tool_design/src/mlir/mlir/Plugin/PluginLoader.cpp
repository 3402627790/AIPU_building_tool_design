//===- PluginLoader.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the relevant interfaces of the plugin.
//
//===----------------------------------------------------------------------===//

#include "Plugin/PluginLoader.h"
#include "Plugin/Plugin.h"
#include "utils/utils.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <gcompiler_api.h>
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace aipugc {

#define PATH_SPLITTER "/"
#define DLIB_SUFFIX ".so"
#ifdef _WIN32
#define PATH_SPLITTER "\\"
#define DLIB_SUFFIX ".dll"
#endif
#define DEFAULT_LIB_PATH "Plugin"
#define AIPUGC_PLUGIN_PATH "AIPUGC_PLUGIN_PATH"
static std::map<std::string, PassFunc> allPluginContexts;
void AipuRegisterPlugin(PluginContext *plugin) {
  if (!allPluginContexts.count(plugin->pass_name)) {
    allPluginContexts[plugin->pass_name] = plugin->pass_func;
  }
}

static inline bool endsWith(const std::string &s, const std::string &suffix) {
  return s.rfind(suffix) == (s.size() - suffix.size()) &&
         s.size() >= suffix.size();
}

PluginLoader::PluginLoader() {

  // 1. use cwd as prior
  loadAllPlugin(DEFAULT_LIB_PATH);

  // 2. third env variable
  if (!pluginPath.empty()) {
    for (auto p : utils::split_string(pluginPath.c_str(), ':')) {
      loadAllPlugin(p.c_str());
    }
  }
  // 3. second env variable
  char *envPluginPath = getenv(AIPUGC_PLUGIN_PATH);
  if (envPluginPath) {
    for (auto p : utils::split_string(envPluginPath, ':')) {
      loadAllPlugin(p.c_str());
    }
  }
  // 4. default place as third
  auto plugin_path = utils::get_exe_dir() + "/Plugin";
  loadAllPlugin(plugin_path.c_str());
}

PluginLoader::~PluginLoader() {
  for (auto plugin : handles) {
    if (plugin.second) {
      llvm::dbgs() << "Unload plugin: " << plugin.first << "\n";
      if (dlclose(plugin.second) != 0) {
        dlerror();
      }
    }
  }
}

void PluginLoader::loadAllPlugin(const char *libPath) {
  DIR *dir;
  struct dirent *ent;
  if (utils::path_exists(libPath)) {
    if ((dir = opendir(libPath)) != NULL) {
      while ((ent = readdir(dir)) != NULL) {
        std::string filename = ent->d_name;
        if (endsWith(filename, DLIB_SUFFIX)) {
          filename = std::string(libPath) + PATH_SPLITTER + filename;
          llvm::dbgs() << "Load " << filename << "\n";
          loadPlugin(filename);
        }
      }
      closedir(dir);
    }
  }
}

void PluginLoader::loadPlugin(const std::string &filename) {
  if (handles.count(filename)) {
    // loaded~~
    return;
  }
  void *handle = nullptr;
  handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == NULL) {
    llvm::dbgs() << " Load plugin failed: " << dlerror() << "\n";
  } else {
    handles[filename] = handle;
  }
}

PluginLoader *PluginLoader::getInstance() {
  static PluginLoader singleton;
  return &singleton;
}

std::vector<PassFunc> PluginLoader::pluginPasses() {
  std::vector<PassFunc> passes;
  for (auto &kv : allPluginContexts) {
    passes.push_back(kv.second);
  }
  return passes;
}

bool PluginLoader::verity() {
  if (pluginPasses().size() > 0)
    return true;
  return false;
}

void AipuApplyPassPlugin(mlir::OpPassManager &optPM) {
  auto *pluginloader = PluginLoader::getInstance();
  if (!pluginloader->verity())
    return;

  for (auto pass : pluginloader->pluginPasses()) {
    optPM.addPass(pass());
  }
  return;
}

}; // namespace aipugc
}; // namespace mlir
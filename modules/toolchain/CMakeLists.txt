project(AIPUToolChain)

set(LibName "aiputoolchain")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "-fno-rtti   -fPIC")

set(ENV{PATH} "${LLVM_PATH}/bin:${ENV}{PATH}")

option (USE_MULTIPROCESS "Use provided math implementation" ON)
if (USE_MULTIPROCESS)
    add_definitions(-DMULTIPROCESS)
endif(USE_MULTIPROCESS)

# find_package(LLVM)
include_directories("${ROOT_DIR}/include")

include_directories("${LLVM_PATH}/include" )
link_directories(${LLVM_PATH}/lib)

add_library(${LibName} SHARED
  wrapper.cpp
)

# set_target_properties( ${LibName} PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.script)

add_executable(test
  test.cpp
)

target_link_libraries(${LibName}
  aipubuildtool
  AIPUOGB
  # lldELF
  # lldCommon
  # LLVMSupport
  # LLVMOption
  # LLVMObject
  # LLVMDebugInfoDWARF
  # LLVMAIPU2AsmParser
  # LLVMAIPU2Disassembler
  # LLVMAIPU2AsmPrinter
  # LLVMAIPU2CodeGen
  # LLVMLTO
  # -fpie
  # -pie
  # -Wl,-E
  # -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.script
)
target_link_libraries(test
  AIPUOGB
  aipubuildtool
  -fpie
  -pie
  -Wl,-E
)


add_custom_command(
  TARGET   ${LibName}
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/../../src/libaiputoolchain.so &&  ln -s ${CMAKE_CURRENT_BINARY_DIR}/libaiputoolchain.so ${CMAKE_CURRENT_BINARY_DIR}/../../src/
)

install(TARGETS ${LibName}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
        )

project(AIPUBuildPluginLUTExample)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${C_CXX_FLAGS}  -g -std=c11 -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   ${C_CXX_FLAGS} -g  -std=c++17 -Wall -Werror  -fno-exceptions -fPIC -Wsign-promo  -Wno-null-dereference -fvisibility-inlines-hidden -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast -Werror=return-type   -Wno-enum-compare  -Wno-expansion-to-defined  -Wno-multichar -fno-strict-aliasing")

set(AIPUBUILDERROOT "/project/ai/scratch01/AIPUBuilder/")
set(PLUGIN_DIR "${PROJECT_SOURCE_DIR}/plugin")
add_custom_target(LINKAIPURUN ALL)
link_directories(${AIPUBUILDERROOT}/lib)
include_directories("${AIPUBUILDERROOT}/include")
# include_directories("${PROJECT_SOURCE_DIR}/../../include")
aux_source_directory("." PLUGIN_SRC)
add_library(lutplugin SHARED ${PLUGIN_SRC})
target_link_libraries(lutplugin aipubuildtool dl)
add_custom_command(
  TARGET   LINKAIPURUN
  COMMAND  rm -rf  ${CMAKE_CURRENT_BINARY_DIR}/bin &&
  mkdir -p  ${CMAKE_CURRENT_BINARY_DIR}/bin 
  &&  ln -s ${AIPUBUILDERROOT}/bin/* ${CMAKE_CURRENT_BINARY_DIR}/bin
)
add_custom_command(
  TARGET   lutplugin
  COMMAND  rm  -rf ${PLUGIN_DIR} && mkdir -p ${PLUGIN_DIR} && ln -s ${CMAKE_CURRENT_BINARY_DIR}/*.so ${PLUGIN_DIR}
)
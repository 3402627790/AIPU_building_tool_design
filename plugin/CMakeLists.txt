
project(AIPUBuildPlugin)

include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}")
set(AIPUBuildLIB ${AIPUBuildLIB} m aipubuildtool)
set(PLUGIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/../tools/plugin/")

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(ALL_PLUGINS ${PROJECT_SOURCE_DIR})
FOREACH(plugin ${ALL_PLUGINS})
    get_filename_component(toolname ${plugin} NAME)
    set(PLUGIN_SRCS "")
    aux_source_directory(${plugin} PLUGIN_SRCS)
    if(NOT "${PLUGIN_SRCS}" MATCHES "^$")
      add_library(${plugin} SHARED ${PLUGIN_SRCS})
      target_link_libraries(${plugin} ${AIPUBuildLIB})
      add_dependencies(${plugin} aipubuildtool)
      install(TARGETS ${plugin}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin/plugin
        ARCHIVE DESTINATION lib/static
      )
      add_custom_command(
        TARGET   ${plugin}
        COMMAND  mkdir -p ${PLUGIN_DIR} && rm -rf ${PLUGIN_DIR}/lib${plugin}.so
        COMMAND  cp -s ${BT_PLUGIN_PATH}/lib${plugin}.so ${PLUGIN_DIR} || echo "cannot link BT_PLUGIN_PATH"
      )
    endif()
ENDFOREACH()
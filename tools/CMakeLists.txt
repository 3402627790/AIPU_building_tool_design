
project(AIPUBuildTool)

include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/include")
include_directories("${ROOT_DIR}/modules/btrt/include")

include_directories("${PROJECT_SOURCE_DIR}")
set(AIPUBuildLIB ${AIPUBuildLIB} m aipubuildtool)
set(aipubtep aipubttools)
set(REL_LIB_PATH_FLAG "")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
set(REL_LIB_PATH_FLAG "-Wl,-rpath '-Wl,\$ORIGIN/../lib/'")
endif()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${C_CXX_FLAGS}   -Wall -Werror ${REL_LIB_PATH_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   ${C_CXX_FLAGS}   -Wall -Werror ${REL_LIB_PATH_FLAG}")
include_directories("${ROOT_DIR}/3rdparty")


FILE(GLOB COMMON_SRCS  ${PROJECT_SOURCE_DIR}/common/**.cpp)
add_library(${aipubtep} SHARED ${COMMON_SRCS})
target_link_libraries(${aipubtep} ${AIPUBuildLIB} aipubtrt )


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

SUBDIRLIST(ALL_TOOLS ${PROJECT_SOURCE_DIR})
list(REMOVE_ITEM ALL_TOOLS "common")
FOREACH(tool ${ALL_TOOLS})
    get_filename_component(toolname ${tool} NAME)
    set(TOOL_SRCS "")
    aux_source_directory(${tool} TOOL_SRCS)
    add_executable(${toolname} ${TOOL_SRCS})
    target_link_libraries(${toolname} ${aipubtep} ${AIPUBuildLIB})
    add_dependencies(${toolname} aipubuildtool)
    install(TARGETS ${toolname}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
        )
ENDFOREACH()
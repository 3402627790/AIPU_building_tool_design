
project(AIPUBuildToolPythonAPI)
set(REL_LIB_PATH_FLAG "")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
set(REL_LIB_PATH_FLAG "-Wl,-rpath '-Wl,\$ORIGIN'  -Wl,-z,origin")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions  -fPIC  ${REL_LIB_PATH_FLAG}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions  -fPIC ${REL_LIB_PATH_FLAG}")


include_directories("${ROOT_DIR}/src")
include_directories("${ROOT_DIR}/include")
include_directories("${ROOT_DIR}/tools")
include_directories("${ROOT_DIR}/3rdparty")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}")

find_package(Threads REQUIRED)
FILE(GLOB_RECURSE  PYTHON_SRCS ${PROJECT_SOURCE_DIR}/CGBuilder/**.py )


if(OS_ANDROID)
  message("python not support android build")
else() 
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/src/AIPUBuilder" 
    DESTINATION python/src/
    PATTERN "*/bin/*"
    PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_READ WORLD_EXECUTE
  )
  add_subdirectory(Wrapper)
endif()

project(PyBTRT CXX C)
set(PyAPI "_btrt")

set(PYTHON_DES "${ROOT_DIR}/python/src/AIPUBuilder/_C")

include_directories("${PROJECT_SOURCE_DIR}/../src")

FILE(GLOB PyBTRT_SRCS ${PROJECT_SOURCE_DIR}/csrc/**.cpp)

# for #include <python.h>
execute_process (
    COMMAND ${PYTHON_EXE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDES
    RESULT_VARIABLE CMD_RESULT
)
if(NOT CMD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to execute Python script")
endif()
include_directories(${PYTHON_INCLUDES})


add_library(${PyAPI}  SHARED  ${PyBTRT_SRCS})
SET_TARGET_PROPERTIES(${PyAPI} PROPERTIES PREFIX   "")
target_link_libraries(${PyAPI}  PUBLIC ${LibName})
add_custom_command(
        TARGET   ${PyAPI}
        COMMAND  rm -rf ${PYTHON_DES}/${PyAPI}.so
        COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/${PyAPI}.so ${PYTHON_DES}/
        DEPENDS ${PyAPI}
)

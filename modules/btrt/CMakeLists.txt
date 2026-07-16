project(BTRT CXX C)

set(LibBTRT "aipubtrt")
set(LibBTSIMDEV "aipubtsimdev")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${ROOT_DIR}/3rdparty")
include_directories("${ROOT_DIR}/include")
include_directories("${ROOT_DIR}/modules/btrt/include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

FILE(GLOB COMMON_SRCS  ${PROJECT_SOURCE_DIR}/src/common/**.cpp)
FILE(GLOB SIMDEV_SRCS ${PROJECT_SOURCE_DIR}/src/sim_dev/**.cpp )
add_library(${LibBTSIMDEV} SHARED ${SIMDEV_SRCS} ${COMMON_SRCS})
target_link_libraries(${LibBTSIMDEV} dl)

FILE(GLOB BTRT_SRCS ${PROJECT_SOURCE_DIR}/src/rt/**.cpp)
add_library(${LibBTRT} SHARED ${BTRT_SRCS} ${COMMON_SRCS})
target_link_libraries(${LibBTRT} dl)

add_custom_command(
  TARGET   ${LibBTRT} ${LibBTSIMDEV}
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/../../src/lib${LibBTRT}.so
  COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${LibBTRT}.so ${CMAKE_CURRENT_BINARY_DIR}/../../src/
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/../../src/lib${LibBTSIMDEV}.so
  COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${LibBTSIMDEV}.so ${CMAKE_CURRENT_BINARY_DIR}/../../src/
  COMMAND  rm -rf ${PYTHON_DES}/lib${LibBTRT}.so
  COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${LibBTRT}.so  ${PYTHON_DES}/
  COMMAND  rm -rf ${PYTHON_DES}/lib${LibBTSIMDEV}.so
  COMMAND  ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${LibBTSIMDEV}.so  ${PYTHON_DES}/
)

install(TARGETS ${LibBTRT} ${LibBTSIMDEV}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
        )
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include/modules/btrt)

if(OS_ANDROID)
  message("python not support android build")
else()
  add_subdirectory(python)
endif()

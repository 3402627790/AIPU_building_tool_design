project(BTRT CXX C)

set(LibName "aipubtrt")

set(x5_simu_name "aipu_simulator_x5_ft")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${ROOT_DIR}/3rdparty")
include_directories("${X5SIMU_PATH}/include")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
link_directories(${X5SIMU_PATH}/lib)


FILE(GLOB_RECURSE  BTRT_SRCS ${PROJECT_SOURCE_DIR}/src/**.cpp )

add_library(${LibName} SHARED ${BTRT_SRCS})


target_link_libraries(${LibName}
  ${x5_simu_name}
)

add_custom_command(
  TARGET   ${LibName}
  COMMAND  rm -rf ${CMAKE_CURRENT_BINARY_DIR}/../../src/lib${LibName}.so &&  ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib${LibName}.so ${CMAKE_CURRENT_BINARY_DIR}/../../src/
)

install(TARGETS ${LibName}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
        )
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include/modules/btrt)

add_subdirectory(python)

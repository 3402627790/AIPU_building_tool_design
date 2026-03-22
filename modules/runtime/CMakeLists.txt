project(AIPURuntimeZ5)
set(LibName "aipubtdriver")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories("${ROOT_DIR}/include")
include_directories("${DRIVER_API}")

link_directories(${SIM_PATH})
link_directories(${DRIVER_PATH})

add_library(${LibName} SHARED
  ${PROJECT_SOURCE_DIR}/wrapper.cpp
  ${PROJECT_SOURCE_DIR}/stub.cpp
)


target_link_libraries(${LibName}
aipudrv
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
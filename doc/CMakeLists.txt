
project(AIPUBuildToolDocs)


set(SPHINXBUILD "sphinx-build" CACHE STRING "Path to sphinx-build executable")



FILE(GLOB_RECURSE  DOC_SRCS
    ${PROJECT_SOURCE_DIR}/**.rst
    ${PROJECT_SOURCE_DIR}/**md
    ${PROJECT_SOURCE_DIR}/conf.py
)

add_custom_target(
    docs 
    COMMAND ${CMAKE_COMMAND} -E echo "Building documentation..."
    COMMAND echo "${SPHINXBUILD} -M html ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}"
    COMMAND ${SPHINXBUILD} -M html "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}"
  DEPENDS ${DOC_SRCS}
)

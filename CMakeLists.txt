cmake_minimum_required(VERSION 3.16)

project(HunyuanOCRncnn VERSION 0.1.0 LANGUAGES CXX C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(HUNYUAN_OCR_BUILD_CLI "Build the hunyuan-ocr command-line application" ON)
option(HUNYUAN_OCR_USE_SYSTEM_NCNN "Use an installed ncnn package" OFF)

if(HUNYUAN_OCR_USE_SYSTEM_NCNN)
    find_package(ncnn REQUIRED)
else()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
        message(FATAL_ERROR
            "The ncnn submodule is not initialized. Run: "
            "git submodule update --init --recursive"
        )
    endif()
    set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ncnn)
endif()
find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(hunyuan_ocr_ncnn_runtime
    src/runtime.cpp
    src/internal/runtime_impl.cpp
    src/internal/text_runtime.cpp
    src/internal/sampling.cpp
    src/internal/image_preprocessor.cpp
    src/internal/image.cpp
    src/internal/xdrope.cpp
    src/internal/tokenizer/bpe_tokenizer.cpp
)
add_library(HunyuanOCRncnn::runtime ALIAS hunyuan_ocr_ncnn_runtime)

set_target_properties(hunyuan_ocr_ncnn_runtime PROPERTIES
    EXPORT_NAME runtime
    OUTPUT_NAME hunyuan_ocr_ncnn
    POSITION_INDEPENDENT_CODE ON
)

target_compile_features(hunyuan_ocr_ncnn_runtime PUBLIC cxx_std_17)
target_include_directories(hunyuan_ocr_ncnn_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(hunyuan_ocr_ncnn_runtime SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(hunyuan_ocr_ncnn_runtime
    PRIVATE ncnn Threads::Threads nlohmann_json::nlohmann_json
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hunyuan_ocr_ncnn_runtime PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(hunyuan_ocr_ncnn_runtime PRIVATE /W4)
endif()

if(HUNYUAN_OCR_BUILD_CLI)
    add_executable(hunyuan_ocr_cli apps/hunyuan_ocr_cli.cpp)
    set_target_properties(hunyuan_ocr_cli PROPERTIES OUTPUT_NAME hunyuan-ocr)
    target_link_libraries(hunyuan_ocr_cli PRIVATE HunyuanOCRncnn::runtime nlohmann_json::nlohmann_json)
endif()

if(BUILD_TESTING)
    add_executable(runtime_core_test tests/cpp/runtime_core_test.cpp)
    target_include_directories(runtime_core_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(runtime_core_test PRIVATE HunyuanOCRncnn::runtime ncnn)
    add_test(NAME runtime_core COMMAND runtime_core_test)

    foreach(tool IN ITEMS vision_replay image_vision_replay lm_head_replay benchmark)
        add_executable(${tool} tests/cpp/${tool}.cpp)
        target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(${tool} PRIVATE HunyuanOCRncnn::runtime ncnn nlohmann_json::nlohmann_json)
    endforeach()

    find_program(PYTEST_EXECUTABLE NAMES pytest PATHS ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin NO_DEFAULT_PATH)
    if(NOT PYTEST_EXECUTABLE)
        find_program(PYTEST_EXECUTABLE NAMES pytest)
    endif()
    if(PYTEST_EXECUTABLE)
        add_test(
            NAME python_tests
            COMMAND ${PYTEST_EXECUTABLE} -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/python
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

install(
    TARGETS hunyuan_ocr_ncnn_runtime
    EXPORT HunyuanOCRncnnTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(HUNYUAN_OCR_BUILD_CLI)
    install(TARGETS hunyuan_ocr_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRncnnConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    cmake/HunyuanOCRncnnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRncnnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCRncnn
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRncnnConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRncnnConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCRncnn
)
install(
    EXPORT HunyuanOCRncnnTargets
    FILE HunyuanOCRncnnTargets.cmake
    NAMESPACE HunyuanOCRncnn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCRncnn
)

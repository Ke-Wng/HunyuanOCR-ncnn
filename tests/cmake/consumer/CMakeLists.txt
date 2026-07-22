cmake_minimum_required(VERSION 3.16)
project(HunyuanOCRncnnConsumer LANGUAGES CXX)

find_package(HunyuanOCRncnn 0.1 REQUIRED)

add_executable(hunyuan_ocr_consumer main.cpp)
target_link_libraries(hunyuan_ocr_consumer PRIVATE HunyuanOCRncnn::runtime)

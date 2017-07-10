cmake_minimum_required(VERSION 3.2)
project(sequencial)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
	rules.h
		rules.cpp
		stemmer.cpp)

add_executable(sequencial ${SOURCE_FILES})








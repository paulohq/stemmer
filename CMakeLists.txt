cmake_minimum_required(VERSION 3.7)
project(stemmer)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        arquivos/lista-palavras-ingles.txt
        le_arquivo.cpp
        sequencial/rules.cpp
        sequencial/rules.h
        sequencial/rules_inverso.cpp
        stemmer.c
        sequencial/stemmer.cpp
        sequencial/stemmer_inverso.cpp
        paralelo/stemmer_inverso_paralelo.cpp
        teste_le_arq.cpp paralelo/kernel_stemmer.cu paralelo/myruntime.cu)

add_executable(stemmer ${SOURCE_FILES})
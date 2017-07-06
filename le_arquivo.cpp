#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>

int main(int argc, char *argv[]) {
    char filename[] = "arquivos/lista-palavras-ingles.txt";
    char *buffer = 0;
    long length;
    FILE *f = fopen(filename, "r");

    if (f) {
        fseek(f, 0, SEEK_END);
        length = ftell(f);
        fseek(f, 0, SEEK_SET);
        //std::malloc caso use <cstdlib>
        buffer = (char*)malloc(length);
        if (buffer) {

            fread(buffer, 1, length, f);
        }
        fclose(f);
    }

    if (buffer) {
        unsigned long int num = sizeof(buffer);
        printf("%lu\n\n", num);
        printf("%s\n\n", buffer);
    }
}
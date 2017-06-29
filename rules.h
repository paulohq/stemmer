#include <stdio.h>
#include <stdlib.h>

#define QUANTIDADE_REGRAS 116
struct Regra {
        char sufixo[10];
        char asterisco[2];
        int qtde;
        char rep[3];
        char final[2];
};

extern struct Regra regras[QUANTIDADE_REGRAS];

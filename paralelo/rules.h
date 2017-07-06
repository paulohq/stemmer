#include <stdio.h>
#include <stdlib.h>

#define QUANTIDADE_REGRAS 116
#define TAMANHO_PALAVRA 64
#define BILLION  1000000000L
struct Regra {
        char sufixo[10];
        char asterisco[2];
        int qtde;
        char rep[3];
        char final[2];
};

extern struct Regra regras[QUANTIDADE_REGRAS];

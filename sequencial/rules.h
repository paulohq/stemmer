#include <stdio.h>
#include <stdlib.h>

//Quantidade de regras no vetor de regras.
#define QUANTIDADE_REGRAS 116
//Tamanho máximo da palavra.
#define TAMANHO_PALAVRA 64

#define BILLION  1000000000L
/**
 * Estrutura que guarda as regras que serão usadas para processar o stem
 * sufixo: o sufixo que será comparado com o final da palavra.
 * sufixo_repete: indica se o sufixo repete ou não.
 *                * - sufixo repete
 *                vazio - sufixo não repete
 * qtde: quantidade de caracteres que serão retirados do final da palavra. Pode ser diferente do tamanho do sufixo
 *       porque nem sempre será retirado o sufixo completo da palavra.
 * final: indica se o processamento do sufixo deve continuar para tentar retirar outro sufixo ou não. Ou seja, uma
 *        mesma palavra pode ter mais de um sufixo retirado.
 *        . - indica que não deve continuar
 *        > - indica que pode continuar
 */
struct Regra {
        char sufixo[10];
        char sufixo_repete[2];
        int qtde;
        char rep[3];
        char final[2];
};

extern struct Regra regras[QUANTIDADE_REGRAS];

/**
 * Trabalho da matéria de PPD
 * Paulo Henrique da Silva
 *
 *
 * Definição da estrutura de regras para o stemmer de Paice/Husk.
 */

#include <stdio.h>
#include <stdlib.h>

#define BILLION  1000000000L

//Tamanho máximo da palavra.
#define TAMANHO_PALAVRA 64
//Quantidade de regras no vetor de regras.
#define QUANTIDADE_REGRAS 120
/**
 * Estrutura que guarda as regras que serão usadas para processar o stem
 * sufixo:        o sufixo que será comparado com o final da palavra.
 * qtde_retirada: quantidade de caracteres que serão retirados do final da palavra. Pode ser diferente do tamanho
 *                do sufixo porque nem sempre será retirado o sufixo completo da palavra.
 * rep:           caracteres que serão adicionados ao final da palavra (pode ser vazio).
 * final:         indica se o processamento do sufixo deve continuar para tentar retirar outro sufixo ou não. Ou seja, uma
 *                mesma palavra pode ter mais de um sufixo retirado.
 *               . - indica que não deve continuar
 *               > - indica que pode continuar
 */
struct Regra {
        char sufixo[10];
        int qtde_retirada;
        char rep[3];
        char final[2];
};

extern struct Regra regras[QUANTIDADE_REGRAS];

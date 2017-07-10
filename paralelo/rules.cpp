/**
 * Trabalho da matéria de PPD
 * Paulo Henrique da Silva
 *
 *
 * Codificação da estrutura de regras para o stemmer de Paice/Husk.
 */

#include "rules.h"

/**
 * Vetor de estrutura que guarda as regras para o processamento do stem.
 * sufixo:        o sufixo que será comparado com o final da palavra.
 * qtde_retirada: quantidade de caracteres que serão retirados do final da palavra. Pode ser diferente do tamanho
 *                do sufixo porque nem sempre será retirado o sufixo completo da palavra.
 * rep:           caracteres que serão adicionados ao final da palavra (pode ser vazio).
 * final:         indica se o processamento do sufixo deve continuar para tentar retirar outro sufixo ou não. Ou seja, uma
 *                mesma palavra pode ter mais de um sufixo retirado.
 *               . - indica que não deve continuar
 *               > - indica que pode continuar
 */
struct Regra regras[QUANTIDADE_REGRAS] =  {
		{"a", 1, "", "."},
		{"ae", 2,"","."},
		{"ai", 2, "", "."},
		{"ais", 3, "", "."},
		{"ata", 3, "", "."},
		{"bb", 1, "", "."},
		{"city", 3, "s", ">"},
		{"ci", 2, "", ">"},
		{"cn", 1, "t", ">"},
		{"dd", 1, "", "."},
		{"dei",  3, "y", ">"},
		{"deec", 2, "ss", "."},
		{"dee", 1, "", "."},
		{"de", 2, "", ">"},
		{"dooh", 4, "", ">"},
		{"e", 1, "", ">"},
		{"feil", 1, "v", "."},
		{"fi", 2, "", ">"},
		{"gni", 3, "", ">"},
		{"gai", 3, "y", "."},
		{"ga", 2, "", ">"},
		{"gg", 1, "", "."},
		{"ht", 2, "", "."},
		{"hsiug", 5, "ct", "."},
		{"hsi", 3, "", ">"},
		{"i", 1, "", "."},
		{"i", 1, "y", ">"},
		//{"@i", "", 1, "d", "."},
		{"juf", 1, "s", "."},
		{"ju", 1, "d", "."},
		{"jo", 1, "d", "."},
		{"jeh", 1, "r", "."},
		{"jrev", 1, "t", "."},
		{"jsim", 2, "t", "."},
		{"jn", 1, "d", "."},
		{"j", 1, "s", "."},
		{"lbaifi", 6, "", "."},
		{"lbai", 4, "y", "."},
		{"lba", 3, "", ">"},
		{"lbi", 3, "", "."},
		{"lib", 2, "l", ">"},
		{"lc", 1, "", "."},
		{"lufi", 4, "y", "."},
		{"luf", 3, "", ">"},
		{"lu", 2, "", "."},
		{"lai", 3, "", ">"},
		{"lau", 3, "", ">"},
		{"la", 2, "", ">"},
		{"ll", 1, "", "."},
		{"mui", 3, "", "."},
		{"mu", 2, "", "."},
		{"msi", 3, "", ">"},
		{"mm", 1, "", "."},
		{"nois", 4, "", ">"},
		{"noix", 4, "ct", "."},
		{"noi", 3, "", ">"},
		{"nai", 3, "", ">"},
		{"na", 2, "", ">"},
		{"nee", 0, "", "."},
		{"ne", 2, "", ">"},
		{"nn", 1, "", "."},
		{"pihs", 4, "", ">"},
		{"pp", 1, "", "."},
		{"re", 2, "", ">"},
		{"rae", 0, "", "."},
		{"ra", 2, "", "."},
		{"ro", 2, "", ">"},
		{"ru", 2, " ", ">"},
		{"rr", 1, "", "."},
		{"rt", 1, "", ">"},
		{"rei", 3, "y", ">"},
		{"sei", 3, "y", ">"},
		{"sis", 2, "", "."},
		{"si", 2, "", ">"},
		{"ssen", 4, "", ">"},
		{"snoiss", 4, "ss","."},
		{"ss", 0, "", "."},
		{"suo", 3, "", ">"},
		{"su", 2, "", "."},
		{"s", 1, "", ">"},
		{"s", 0, "", "."},
		{"tacilp", 4, "y", "."},
		{"ta", 2, "", ">"},
		{"tnem", 4, "", ">"},
		{"tne", 3, "", ">"},
		{"tna", 3, "", ">"},
		{"tpir", 2, "b", "."},
		{"tpro", 2, "b", "."},
		{"tcud", 1, "", "."},
		{"tpmus", 2, "", "."},
		{"tpec", 2, "iv", "."},
		{"tulo", 2, "v", "."},
		{"tsis", 0, "", "."},
		{"tsi", 3, "", ">"},
		{"tt", 1, "", "."},
		{"uqi", 3, "", "."},
		{"ugo", 1, "", "."},
		{"vis", 3, "", ">"},
		{"vie",  0, "", "."},
		{"vi", 2, "", ">"},
		{"ylb", 1, "", ">"},
		{"yli", 3, "y", ">"},
		{"ylp", 0, "", "."},
		{"yl", 2, "", ">"},
		{"ygo", 1, "", "."},
		{"yhp", 1, "", "."},
		{"ymo", 1, "", "."},
		{"ypo", 1, "", "."},
		{"yti", 3, "", ">"},
		{"yte", 3, "", ">"},
		{"ytl",  2, "", "."},
		{"yrtsi", 5, "", "."},
		{"yra", 3, "", ">"},
		{"yro", 3, "", ">"},
		{"yfi", 3, "", "."},
		{"ycn", 2, "t", ">"},
		{"yca", 3, "", ">"},
		{"zi", 2, "", ">"},
		{"zy", 1, "s", "."},
		{"end0", 0, "", ""}

};
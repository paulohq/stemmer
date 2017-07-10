/**
 * Trabalho da matéria de PPD
 * Paulo Henrique da Silva
 *
 *
 * Programa principal que lê um documento da disco para a memória da CPU e depois chama o programa kernel_stemmer.cu
 * para fazer o processamento na GPU.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "rules.h"
#include <time.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>


//nvcc -c -arch=sm_30 kernel_stemmer.cu
//nvcc -arch=sm_30 kernel_stemmer.o stemmer_inverso_paralelo.cpp -o paralelo
// ./paralelo nome_arquivo.txt 0 ou 1 (0 - nao imprime na tela; 1 - imprime na tela)
// as duas formas gravam no arquivo texto saida.txt no mesmo diretório do programa.

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
    int qtde_retirada;
    char rep[3];
    char final[2];
};

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
        {"ciso", 4, "", "."},
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

extern void cuda_stemmer(char *buffer, int *ptr, int numthreads, int tamanhoarquivo, Regra *regras, char * imprime);

/**
 * Processa o buffer (arquivo contendo as palavras) para poder separar as palavras para serem passadas para o kernel fazer o processamento do stem.
 * É armazenado o início de cada palavra no *o_ptr. Sendo 0 para a primeira palavra e depois percorrido o buffer até encontrar o \n
 * A posição do \n + 1 será o início da próxima palavra e assim sucessivamente até o final do buffer.
 * @param filename nome do arquivo que será lido
 * @param o_buffer ponteiro para o buffer
 * @param o_ptr ponteiro para o vetor onde armazena o início de cada palavra no buffer
 * @param o_numpalavras quantidade de palavras encontradas no arquivo
 * @param o_tamanhoarquivo tamanho do arquivo em bytes
 * @return
 */
bool le_arquivo(char *filename, char **o_buffer, int **o_ptr, int *o_numpalavras, int *o_tamanhoarquivo)
{
	int file = 0;
	if((file = open(filename, O_RDONLY)) < 0)
		return false;

	struct stat fileStat;
	if(fstat(file,&(fileStat)) < 0)
		return false;

	//Retorna o tamanho do arquivo para ser usado para alocar memória para a variável buffer.
	size_t tamanhoarquivo = fileStat.st_size;


	//armazena o arquivo inteiro na memória (será lido abaixo).
	char *buffer = new char [tamanhoarquivo + 1];
	//vetor que armazena um índice para a primeira posição de cada palavra no buffer.
	int *ptr = new int [tamanhoarquivo];

	// Lê o arquivo inteiro para a memória (buffer).
	if (read (file, (void *) buffer, tamanhoarquivo) < tamanhoarquivo)
	{
		close (file);
		return false;
	}

	//Atribui 0 para a primeira posição do vetor da primeira palavra encontrada no buffer.
	ptr [0] = 0;
	int j = 1;

	//Laço para percorrer o arquivo e substituir um \n por \0 para indicar o fim de cada palavra.
	//Logo depois de cada \0 inicia uma nova palavra então essa posição + 1 é armazenada no ponteiro ptr
	//para indicar a posição onde começa cada palavra no buffer. Assim é possível marcar o início de cada
	//palavra no buffer e com o \0 o seu final.
	for (int i = 0; i < tamanhoarquivo; i++)
	{
		if (buffer[i] == (char) '\n')
		{
			buffer [i] = (char) '\0';
			ptr [j++] = i + 1;
		}
	}


	*o_buffer = buffer;
	*o_ptr = ptr;
	*o_numpalavras = j;
	*o_tamanhoarquivo = tamanhoarquivo;
}

int main(int argc, char **argv)
{
	//Se não tiver o segundo argumento, pede para informar o nome do arquivo.
	if(argc != 3)
		return printf("%s\n", "Informe o nome do arquivo e a indicacao de imprimir ou nao na tela (0 - nao imprime; 1 - imprime).");


	//variável que armazena os dados do arquivo
	char *buffer;
	//vetor que contém a posição de início de cada palavra do buffer.
	int *ptr;
	//número de palavras que foram encontradas no arquivo
	int numpalavras;
	//tamanho do arquivo em bytes.
	int tamanhoarquivo;

	//chama rotina para processar o arquivo
	le_arquivo(argv[1], &(buffer), &(ptr), &(numpalavras), &tamanhoarquivo);

	cuda_stemmer(buffer, ptr, numpalavras, tamanhoarquivo, regras, argv[2]);


	delete buffer;
	delete ptr;

	return 0;
}
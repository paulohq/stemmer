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


//nvcc -c kernel_stemmer.cu
//nvcc kernel_stemmer.o stemmer_inverso_paralelo.cpp

extern void cuda_stemmer(char **buffer, int *ptr, int numthreads, int tamanhoarquivo);

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

	//printf("File Size: \t\t%ld bytes\n",fileStat.st_size);
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
	if(argc != 2)
		return printf("%s\n", "Informe o nome do arquivo.");

	struct timespec start, stop;
	double accum;

	//marca o início do tempo de processamento.
	clock_gettime( CLOCK_REALTIME, &start);

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

	cuda_stemmer(&buffer, ptr, numpalavras, tamanhoarquivo);

	//marca o final do tempo de processamento.
	clock_gettime( CLOCK_REALTIME, &stop);

	accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / BILLION;
	printf( "%lf\n", accum );

	delete buffer;
	delete ptr;

	return 0;
}
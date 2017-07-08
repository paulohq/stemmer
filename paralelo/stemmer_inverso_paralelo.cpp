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

extern void cuda_stemmer(char *buffer, int *ptr, int numthreads, int tamanhoarquivo);

//Processa o buffer (arquivo contendo as palavras) para poder separar palavra por palavra para serem passadas
//para o kernel fazer o processamento do stem.
//É armazenado o início de cada palavra no *o_ptr. Sendo 0 para a primeira palavra e depois percorrido o buffer até encontrar o \n
//A posição do \n + 1 será o início da próxima palavra e assim sucessivamente até o fim do buffer.
//Recebe o nome do arquivo para ser processado e retorna um poteiro para o buffer
bool le_arquivo(char *filename, char **o_buffer, int **o_ptr, int *o_numpalavras, int *o_tamanhoarquivo)
{
	int file = 0;
	if((file = open(filename, O_RDONLY)) < 0)
		return false;

	struct stat fileStat;
	if(fstat(file,&(fileStat)) < 0)
		return false;

	size_t tamanhoarquivo = fileStat.st_size;

	//printf("File Size: \t\t%ld bytes\n",fileStat.st_size);
	char *buffer = new char [tamanhoarquivo + 1];
	int *ptr = new int [tamanhoarquivo];

	// Lê o arquivo inteiro para a memória.
	if (read (file, (void *) buffer, tamanhoarquivo) < tamanhoarquivo)
	{
		close (file);
		return false;
	}

	//Atribui 0 como a primeira posição da primeira palavra encontrada no buffer.
	ptr [0] = 0;
	int j = 1;

	for (int i = 0; i < tamanhoarquivo; i++)
	{
		if (buffer[i] == (char) '\n')
		{
			buffer [i] = (char) '\0';
			ptr [j++] = i + 1;
		}
	}
/*	for (int i = 0; i < 10; i++)
	{
		printf("ptr=>%d  i=>%d\n", ptr[i], i);
	}*/
	*o_buffer = buffer;
	*o_ptr = ptr;
	*o_numpalavras = j;
	*o_tamanhoarquivo = tamanhoarquivo;
}

int main(int argc, char **argv)
{
	if(argc != 2)
		return printf("%s\n", "Informe o nome do arquivo.");

	struct timespec start, stop;
	double accum;

	clock_gettime( CLOCK_REALTIME, &start);

	char *buffer;
	int *ptr;
	int numpalavras;
	int tamanhoarquivo;
	le_arquivo(argv[1], &(buffer), &(ptr), &(numpalavras), &tamanhoarquivo);

	cuda_stemmer(buffer, ptr, numpalavras, tamanhoarquivo);
	//printf("%d\n", numthreads);

	clock_gettime( CLOCK_REALTIME, &stop);

	accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / BILLION;
	printf( "%lf\n", accum );

	delete buffer;
	delete ptr;

	return 0;
}
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "../sequencial/rules.h"
#include <time.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>


//nvcc -c kernel_stemmer.cu
//nvcc kernel_stemmer.o stemmer_inverso_paralelo.cpp

extern void cuda_stemmer(char *buffer, char **ptr, int numthreads);


bool le_arquivo(char *filename, char **o_buffer, char ***o_ptr, int *o_numthreads)
{
	int file = 0;
	if((file = open(filename, O_RDONLY)) < 0) // -1 é o código de erro. Estava < -1
		return false;

	struct stat fileStat;
	if(fstat(file,&(fileStat)) < 0)
		return false;

	size_t tamanhoArquivo = fileStat.st_size;
	//printf("File Size: \t\t%ld bytes\n",fileStat.st_size);
	char *buffer = new char [tamanhoArquivo + 1];
	char **ptr = new char *[tamanhoArquivo];

	// Lê o arquivo inteiro para a memória.
	if (read (file, (void *) buffer, tamanhoArquivo) < tamanhoArquivo)
	{
		close (file);
		return false;
	}

	ptr [0] = buffer;
	int j = 1;

	for (int i = 0; i < tamanhoArquivo; i++)
	{
		if (buffer[i] == (char) '\n')
		{
			buffer [i] = (char) '\0';
			ptr [j++] = &(buffer [i + 1]);
		}
	}

	*o_buffer = buffer;
	*o_ptr = ptr;
	*o_numthreads = j;
}

int main(int argc, char **argv)
{
	if(argc != 2)
		return printf("%s\n", "Informe o nome do arquivo.");

	struct timespec start, stop;
	double accum;

	clock_gettime( CLOCK_REALTIME, &start);

	char *buffer;
	char **ptr;
	int numthreads;
	le_arquivo(argv[1], &(buffer), &(ptr), &(numthreads));

	cuda_stemmer(buffer, ptr, numthreads);
	//printf("%d\n", numthreads);

	clock_gettime( CLOCK_REALTIME, &stop);

	accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / BILLION;
	printf( "%lf\n", accum );

	delete buffer;
	delete ptr;

	return 0;
}
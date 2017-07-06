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

//nvcc -arch=sm_20 -c file1.cu
//        $ g++ -c file2.cpp
//        $ g++ -o test file1.o file2.o -L/usr/local/cuda/lib64 -lcudart
//        $ ./test
__global__ void stemmer(char *d_out, char *d_in);

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

	// Lê o arquivo todo para a memória.
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

	//printf("%d\n", numthreads);

	// Prepara para chamar kernel.
	const int ARRAY_SIZE = numthreads;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(char);


	//declara ponteiros da GPU.
	char * d_in;
	char * d_out;

	//alocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);

	//transfere o vetor para a GPU.
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	stemmer<<<1, ARRAY_SIZE>>>(d_out, d_in);
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for ( int i = 0; i < ARRAY_SIZE; i++) {
		printf("%s", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);


	clock_gettime( CLOCK_REALTIME, &stop);

	accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / BILLION;
	printf( "%lf\n", accum );

	delete buffer;
	delete ptr;

	return 0;
}
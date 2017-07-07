#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

 __global__ void stemmer(char **in, char **out, int n) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	char buffer [64];

	if (idx < n)
    {
      	char *nav = buffer;
		char *original = in [idx];

		while (*original != '\0')
        {
          	*nav++ = *original++;
        }

      	*nav = '\0';
    }

}

extern void cuda_stemmer(char *buffer, char **ptr, int o_numthreads)
{
    int numberOfBlocks = 2;
    int threadsPerBlock = 5;
    int maxNumberOfThreads = 10;


    //stemmer<<<numberOfBlocks, threadsPerBlock>>>(maxNumberOfThreads);
    //cudaDeviceSynchronize();

    // Prepara para chamar kernel.
    const int ARRAY_SIZE = numthreads;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(char);
    const int TAMANHO_INTEIRO = sizeof(int);


    //declara ponteiros da GPU.
    char * d_in;
    char * d_out;
    int d_numthreads;

    //alocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    cudaMalloc((void **) &numthreads, TAMANHO_INTEIRO);

    //transfere os vetores para a GPU.
    cudaMemcpy(d_in, buffer, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, ptr, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_numthreads, o_numthreads, TAMANHO_INTEIRO, cudaMemcpyHostToDevice);

    stemmer<<<1, ARRAY_SIZE>>>(d_in, d_out, d_numthreads);
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for ( int i = 0; i < ARRAY_SIZE; i++) {
        printf("%s", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

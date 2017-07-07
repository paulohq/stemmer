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

extern void cuda_stemmer(char *buffer, char **ptr, int numthreads)
{
    int numberOfBlocks = 2;
    int threadsPerBlock = 5;
    int maxNumberOfThreads = 10;


    //stemmer<<<numberOfBlocks, threadsPerBlock>>>(maxNumberOfThreads);
    //cudaDeviceSynchronize();

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
}

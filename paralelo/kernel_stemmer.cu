#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "rules.h"
#include "myruntime.cu"


//Retorna a regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada então retorna -1.
__device__ int retorna_regra(char *palavra_reversa, int num_regra) {

    int tamanho;

    //Laço que percorre o vetor de regras.
    for (int i = num_regra; i < QUANTIDADE_REGRAS; i++) {
        // Pega a regra atual do vetor de regras.
        //strcpy(regra, regras[i].sufixo);

        //Pega o sufixo da palavra passada como parametro no mesmo tamanho da regra encontrada acima.
        //É feito para poder comparar o sufixo da palavra reversa passada como parâmetro com a regra do vetor de regras.
        tamanho = strlen(regras[i].sufixo);

        //Compara o sufixo da palavra reversa com a regra. Se forem iguais retorna o número da regra no vetor de regras
        if (strncmp(palavra_reversa, regras[i].sufixo, tamanho) == 0) return i;

    }
    //Se não encontrou nenhuma regra que seja igual ao sufixo da palavra reversa então retorna -1.
    return -1;
}

//Verifica se o caracter passado como parâmetro é vogal ou y.
__device__ bool letra_eh_vogal(char letra) {
    letra = d_tolower(letra);

    if (letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u' || letra == 'y') return true;
    return false;
}

//Verifica se a palavra tem vogal.
__device__ bool string_tem_vogal(char * palavra)
{
    int j;
    for (j = 0; j < strlen(palavra); j++)
    {
        if (letra_eh_vogal(palavra[j])) {
            return true;
        }
    }
    return false;
}


//Verifica se o stem é válido.
__device__ bool valida_stem(char * stem_reverso) {
    //Se o stem inicia com uma vogal então deve ter pelo menos duas letras.
    //ex.: owed ==> ow, owing ==> ow
    //mas não pode ser: ear ==> e

    if (letra_eh_vogal(stem_reverso[0]) == true) {

        if (strlen(stem_reverso) >= 2 )
            return true;
        else
            return false;
    }
    else {
        // Se o stem inicia com uma consoante então deve ter pelo menos três letras.
        if (strlen(stem_reverso) < 3) return false;
        // e pelo menos uma das letras deve ser uma vogal ou y.
        // ex.: crying ==> cry e saying ==> say
        // mas não pode: string ==> str, meant ==> me, cement ==> ce
        if (string_tem_vogal(stem_reverso))
            return true;
        else
            return false;
    }

    return false;
}


//Retorna a string na forma inversa.
__device__ void reverso(char *palavra, char *palavra_reversa) {
    int j, k;

    int len = strlen(palavra);

    for(j = 0, k = len - 1; j < len; j++)
    {
        palavra_reversa[j] = palavra[k];
        k--;
    }

    palavra_reversa[j] = '\0';
}

// retorna o stem (radical) para uma palavra.
__device__ char * stemmer(char *palavra) {

    int intacto = 1;

    char palavra_reversa[TAMANHO_PALAVRA];
    int num_regra = 0;
    char regra[10];
    char vazia[] = "";
    char stem_reverso[TAMANHO_PALAVRA];

    //Chama rotina para colocar a palavra na ordem reversa.
    reverso(palavra, palavra_reversa);

    //Laço que percorre as regras até encontrar um '.' ou 'end0' que indica que não tem mais regras para retirar sufixo.
    while (1) {
        //Chama rotina para tentar encontrar uma regra de remoção/substituição para a palavra.
        //A variável num_regra guarda o índice da última regra encontrada para que seja percorrido no vetor de regras a partir desse índice e não do índice zero do vetor.
        num_regra = retorna_regra(palavra_reversa, num_regra);

        //Se num_regra = -1 então não foi encontrada nenhuma regra para aplicar e o stem foi encontrado.
        if (num_regra == -1) {
            break;
        }
        //copia o sufixo para a variável regra.
        strcpy(regra, regras[num_regra].sufixo);

        //Se
        if ((regras[num_regra].asterisco[0] != '*' || intacto) ) {

            int tamanho_sufixo = strlen(regras[num_regra].sufixo);
            int tamanho = tamanho_sufixo - regras[num_regra].qtde;

            strcpy(stem_reverso, regras[num_regra].rep);
            strncat(stem_reverso, palavra_reversa, tamanho);
            strcat(stem_reverso, &(palavra_reversa[tamanho_sufixo]));

            if (valida_stem(stem_reverso)) {
                strcpy(palavra_reversa, stem_reverso);
                if (regras[num_regra].final[0] == '.') break;
            }
            else {
                //Adicona mais um para ir para próxima regra.
                num_regra++;
            }
        }
        else {
            //Adicona mais um para ir para próxima regra.
            num_regra++;
        }
    }

    //Chama rotina para colocar a palavra na ordem certa pois a mesma foi revertida anteriormente.
    reverso(palavra_reversa, palavra);
    //Retornar ponteiro
    return palavra;

} //end stemmer


__global__ void stemmer(char **buffer, int *ptr, int n, int tamanhobuffer) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    char bufferPalavra [64];

    if (idx < n)
    {
        char *nav = bufferPalavra;
        char *original = buffer [idx];

        while (*original != '\0')
        {
            *nav++ = *original++;
        }

        *nav = '\0';

        stemmer(*nav);
    }

}

extern void cuda_stemmer(char *buffer, int *ptr, int numeropalavras, int tamanhobuffer)
{

    // Prepara para chamar kernel.
    const int ARRAY_SIZE = numeropalavras;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(char *);
    //const int TAMANHO_INTEIRO = sizeof(int);
    //int num_blocos;
    //int num_threads;

    //declara ponteiros da GPU.
    char * d_buffer;
    int * d_ptr;
    //char * d_out;
    //int d_numthreads;
    //int i;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props);

    //int _warpSize = props.warpSize;
    //int _maxThreadsPerBlock = props.maxThreadsPerBlock;

/*
    sSMtoCores nGpuArchCoresPerSM[] =
            {
                    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
                    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
                    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
                    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
                    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
                    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
                    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
                    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
                    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
                    { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
                    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
                    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
                    {   -1, -1 }
            };
*/


    //int _qtdeSM = props.multiProcessorCount;

    //int _maxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;

    //total de cores/ qtde sm = 128
    //128/tam warp = 6
    //maximo threads por sm / 6 = numero_de_threads
    //numthreads / numero_de_threads = numero_blocos
    //arredondar para cima


    //aloca memória na GPU
    cudaMalloc((void **) &d_buffer, tamanhobuffer);
    cudaMalloc((void **) &d_ptr, numeropalavras);

    //transfere os vetores para a GPU.
    cudaMemcpy(d_buffer, buffer, tamanhobuffer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, ptr, ARRAY_BYTES, cudaMemcpyHostToDevice);

    stemmer<<<1, 1>>>(d_buffer, d_ptr, numeropalavras, tamanhobuffer);

    //cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    //for ( int i = 0; i < ARRAY_SIZE; i++) {
    //    printf("%s", h_out[i]);
    //    printf(((i % 4) != 3) ? "\t" : "\n");
    //}

    cudaFree(d_buffer);
    cudaFree(d_ptr);
}
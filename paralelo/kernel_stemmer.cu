/**
 * Trabalho da matéria de PPD
 * Paulo Henrique da Silva
 *
 *
 * Programa para processar os dados do documento na GPU. Tem apenas um kernel que chama os demais procedimentos para
 * para processar o stemming para cada palavra do documento.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include "myruntime.h"


//Quantidade de regras no vetor de regras.
#define QUANTIDADE_REGRAS 120
//Tamanho máximo da palavra.
#define TAMANHO_PALAVRA 64
#define BILLION  1000000000L

struct Regra {
    char sufixo[10];
    int qtde_retirada;
    char rep[3];
    char final[2];
};



//
/**
 * Verifica se o caracter passado como parâmetro é vogal ou y.
 * @param letra letra que será comparada para saber se é vogal ou y.
 * @return
 */
__device__ bool letra_eh_vogal(char letra) {
    letra = d_tolower(letra);

    if (letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u' || letra == 'y') return true;
    return false;
}

//
/**
 * Verifica se a palavra passada como palavra tem alguma vogal.
 * @param palavra
 * @return
 */
__device__ bool string_tem_vogal(char * palavra)
{
    int j;
    for (j = 0; j < d_strlen(palavra); j++)
    {
        if (letra_eh_vogal(palavra[j])) {
            return true;
        }
    }
    return false;
}


/**
 * Verifica se o stem é válido.
 * @param stem_reverso
 * @return
 */
__device__
bool valida_stem(char * stem_reverso) {
    //Se o stem inicia com uma vogal então deve ter pelo menos duas letras.
    //ex.: owed ==> ow, owing ==> ow
    //mas não pode ser: ear ==> e

    if (letra_eh_vogal(stem_reverso[0]) == true) {

        if (d_strlen(stem_reverso) >= 2 )
            return true;
        else
            return false;
    }
    else {
        // Se o stem inicia com uma consoante então deve ter pelo menos três letras.
        if (d_strlen(stem_reverso) < 3) return false;
        // e pelo menos uma das letras deve ser uma vogal ou y.
        // ex.: crying ==> cry e saying ==> say
        // mas não pode: string ==> str, meant ==> me, cement ==> ce
        if (string_tem_vogal(stem_reverso))
            return true;
        else
            return false;
    }

    //return false;
}

/**
 * Retorna a string na forma inversa.
 * @param palavra string na forma normal
 * @param o_palavra_reversa string na forma inversa
 */
__device__ void reverso(char *palavra, char *palavra_reversa) {
    int j, k;

    int len = d_strlen(palavra);

    for(j = 0, k = len - 1; j < len; j++)
    {
        palavra_reversa[j] = palavra[k];
        k--;
    }

    palavra_reversa[j] = '\0';
}



/**
 * Converte string passada como parâmetro para minúsculo.
 * @param palavra string que será convertida
 */
__device__
void minusculo(char *palavra, char *palavra_minusculo) {

    int i;

    for(i = 0; palavra[i]; i++){
        palavra_minusculo[i] = d_tolower(palavra[i]);
    }

    palavra_minusculo[i] = '\0';
}

/**
 * Retorna a regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada então retorna -1.
 * @param palavra_reversa
 * @param num_regra numero da regra para que o laço comece a partir dele (caso já tenha encontrado outra regra antes).
 * @param regras vetor com as regras
 * @return
 */
__device__ int retorna_regra(char *palavra_reversa, int num_regra, Regra *regras) {

    int tamanho;

    //Laço que percorre o vetor de regras.
    for (int i = num_regra; i < QUANTIDADE_REGRAS; i++) {

        //Pega o sufixo da palavra passada como parametro no mesmo tamanho da regra encontrada acima.
        //É feito para poder comparar o sufixo da palavra reversa passada como parâmetro com a regra do vetor de regras.
        tamanho = d_strlen(regras[i].sufixo);

        //Compara o sufixo da palavra reversa com a regra. Se forem iguais retorna o número da regra no vetor de regras
        if (d_strncmp(palavra_reversa, regras[i].sufixo, tamanho)) return i;

    }
    //Se não encontrou nenhuma regra que seja igual ao sufixo da palavra reversa então retorna -1.
    return -1;
}


//
/**
 * Retorna o stem (radical) para uma determinada palavra.
 * @param palavra palavra que será retirado o(s) sufixo(s)
 * @return
 */
__device__
char * stemmer(char *palavra, Regra *regras) {

    char palavra_reversa[TAMANHO_PALAVRA];
    int num_regra = 0;
    char regra[10];
    char stem_reverso[TAMANHO_PALAVRA];

    //Chama rotina para colocar a palavra na ordem inversa.
    reverso(palavra, palavra_reversa);

    //Laço que percorre as regras até encontrar um '.' ou 'end0' que indica que não tem mais regras para retirar sufixo.
    while (1) {

        //Chama rotina para tentar encontrar uma regra de remoção/substituição para a palavra.
        //A variável num_regra guarda o índice da última regra encontrada para que a próxima iteração do laço
        //percorra o vetor de regras a partir do último índice encontrado e não do índice zero do vetor.
        num_regra = retorna_regra(palavra_reversa, num_regra, regras);

        //Se num_regra = -1 então não foi encontrada nenhuma regra para ser aplicada e o stem foi encontrado.
        if (num_regra == -1) {
            break;
        }

        //copia o sufixo para a variável regra.
        d_strncpy(regra, regras[num_regra].sufixo, 10);

        //armazena o tamanho do sufixo.
        int tamanho_sufixo = d_strlen(regras[num_regra].sufixo);
        //armazena a quantidade de caracteres que deverão ser retirados do sufixo.
        int tamanho = tamanho_sufixo - regras[num_regra].qtde_retirada;


        //Os trẽs passos a seguir montam o stem na forma inversa.
        //Recebe os caracteres que serão acrescentados, caso tenha algum, senão será vazio.
        d_strncpy(stem_reverso, regras[num_regra].rep, TAMANHO_PALAVRA);
        //concatena parte da palavra reversa (apenas a quantidade de caracteres da variável tamanho) ao stem reverso.
        d_strncat(stem_reverso, palavra_reversa, tamanho);
        //concatena a palavra reversa a partir da posição tamanho sufixo até o final da palavra ao stem reverso.
        d_strncat(stem_reverso, &(palavra_reversa[tamanho_sufixo]), TAMANHO_PALAVRA);

        /*ex.: connected
               detcennoc
        stem_reverso = "" porque não tem caracteres para serem acrescentados.
        stem_reverso = "" porque a variável tamanho é igual a 0 pois nesse caso é para retirar o sufixo ed completo.
        stem_reverso = "tcennoc" pois concatena a palavra reversa a partir da posição 2 (ou seja retira o sufixo)
        */

        if (valida_stem(stem_reverso)) {
            d_strncpy(palavra_reversa, stem_reverso, TAMANHO_PALAVRA);
            //Se encontrou o sinal de finalização então o stem foi encontrado.
            if (regras[num_regra].final[0] == '.') break;
            //Se encontrou o sinal para continuar volta ao primeiro registro da tabela de regras.
            //Isso é feito para que possa ser aplicada uma nova regra do stemming.
            if (regras[num_regra].final[0] == '>') num_regra = -1;
        } else {
            //Adicona mais um para ir para próxima regra.
            break;
        }

        num_regra++;
    }

    //Chama rotina para colocar a palavra na ordem certa pois a mesma foi revertida anteriormente.
    reverso(palavra_reversa, palavra);

    //Retornar ponteiro
    return palavra;

} //end stemmer





/**
 * Kernel que faz o processamento do arquivo
 * @param buffer contem as palavras do documento.
 * @param ptr ponteiro com o índice onde inicia cada palavra no buffer.
 * @param regras estrutura contendo as regras para processamento dos sufixos.
 * @param n número de palavras que o documento possui (buffer).
 * @param buffer_saida buffer que armazena os stems para serem copiados de volta para a memória
 *                     da CPU para serem gravados no arquivo de saída.
 */
__global__ void kernel_stemmer(char *buffer, int *ptr, Regra *regras, int n, char * buffer_saida) {
    // índice da thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    char bufferPalavra [TAMANHO_PALAVRA];

    if (idx < n)
    {
        //Para cada thread disparada uma palavra do buffer é copiada para a variável original
        char *original = &buffer[ptr[idx]];
        //
        char *saida = &buffer_saida[ptr[idx]];

        //Chama rotina para colocar a palavra em minúsculo.
        minusculo(original, bufferPalavra);
        //Chama rotina que processa o stemming para cada palavra
        stemmer(bufferPalavra, regras);
        //Sincroniza as threads.
        __syncthreads();
        //Copia o stem para o ponteiro saída.
        d_strncpy(saida, bufferPalavra, TAMANHO_PALAVRA);

    }

}

extern void cuda_stemmer(char *buffer, int *ptr, int numeropalavras, int tamanhobuffer, Regra *regras, char * imprime)
{
    struct timespec start, stop;
    double accum;

    // Prepara para chamar kernel.
    const int ARRAY_SIZE = numeropalavras;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int *);

    //declara ponteiros da GPU.
    char * d_buffer;
    int * d_ptr;
    Regra * d_regras;
    char * d_buffer_saida;
    //declara ponteiro na CPU
    char * h_buffer_saida;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props,0);
    int numblocks;
    int numthreads = props.maxThreadsPerBlock;
    size_t tamanho = numeropalavras;

    numblocks = (tamanho + numthreads - 1) / numthreads;


    h_buffer_saida = (char *) malloc(tamanhobuffer);

    //aloca memória na GPU
    cudaMalloc((void **) &d_buffer, tamanhobuffer * sizeof(char));
    cudaMalloc((void **) &d_ptr, ARRAY_BYTES);
    cudaMalloc((void **) &d_regras, QUANTIDADE_REGRAS * sizeof (struct Regra));
    cudaMalloc((void **) &d_buffer_saida, tamanhobuffer * sizeof(char));


    //transfere os vetores para a GPU.
    cudaMemcpy(d_buffer, buffer, tamanhobuffer, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, ptr, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_regras, regras, QUANTIDADE_REGRAS * sizeof (struct Regra), cudaMemcpyHostToDevice);

    //marca o início do tempo de processamento.
    clock_gettime( CLOCK_REALTIME, &start);

    kernel_stemmer<<<numblocks, numthreads>>>(d_buffer, d_ptr, d_regras, numeropalavras, d_buffer_saida);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buffer_saida, d_buffer_saida, tamanhobuffer, cudaMemcpyDeviceToHost );

    //marca o final do tempo de processamento.
    clock_gettime( CLOCK_REALTIME, &stop);

    if ((stop.tv_nsec - start.tv_nsec) < 0)
        accum = (stop.tv_sec - start.tv_sec - 1) + (1000000000 + stop.tv_nsec - start.tv_nsec);
    else
        accum = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
    accum = accum / BILLION;
    printf( "%lf\n", accum );

    FILE *arquivosaida;
    int result;

    // Cria um arquivo texto para gravação.
    arquivosaida = fopen("saida.txt", "w");
    if (arquivosaida == NULL)
        printf("Problemas na criacao do arquivo\n");

    for (int i = 0; i < numeropalavras - 1; i++)
    {
        int pos = ptr[i];
        char *antes = &buffer[pos];
        char *depois = &h_buffer_saida[pos];

        result = fprintf(arquivosaida,"Linha => %d Palavra => %s Stem => %s\n\n", i, antes, depois);
        if (result == EOF)
            printf("Erro na Gravacao\n");

        if (imprime[0] == '1')
          printf("Linha => %d Palavra => %s Stem => %s\n\n", i, antes, depois);
    }

    fclose(arquivosaida);


    cudaFree(d_buffer);
    cudaFree(d_ptr);
    cudaFree(d_regras);
    cudaFree(d_buffer_saida);

    //Reseta a GPU.
    cudaDeviceReset();
}
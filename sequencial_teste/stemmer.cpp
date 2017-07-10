/**
 * Trabalho da matéria de PPD
 * Paulo Henrique da Silva
 *
 *
 * Programa para processar os dados do documento na CPU.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "rules.h"
#include <time.h>


// g++ stemmer_inverso.cpp rules_inverso.cpp -o sequencial
// ./sequencial nome_arquivo.txt 0 ou 1 (0 - nao imprime na tela; 1 - imprime na tela)
// as duas formas gravam no arquivo texto saida.txt no mesmo diretório do programa.


//
/**
 * Verifica se o caracter passado como parâmetro é vogal ou y.
 * @param letra letra que será comparada para saber se é vogal ou y.
 * @return
 */
bool letra_eh_vogal(char letra) {
	letra = tolower(letra);
	
	if (letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u' || letra == 'y') return true;
	return false;
}

//
/**
 * Verifica se a palavra passada como palavra tem alguma vogal.
 * @param palavra
 * @return
 */
bool string_tem_vogal(char * palavra)
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


/**
 * Verifica se o stem é válido.
 * @param stem_reverso
 * @return
 */
bool valida_stem(char * stem_reverso) {
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



/**
 * Retorna a string na forma inversa.
 * @param palavra string na forma normal
 * @param o_palavra_reversa string na forma inversa
 */
void reverso(char *palavra, char *o_palavra_reversa) {
	int j, k;
	
	int len = strlen(palavra);
	
	for(j = 0, k = len - 1; j < len; j++)
	{
		o_palavra_reversa[j] = palavra[k];
		k--;
	}

	o_palavra_reversa[j] = '\0';
}

/**
 * Retorna a regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada então retorna -1.
 * @param palavra_reversa palavra passada na forma reversa.
 * @param num_regra numero da regra para que o laço comece a partir dele (caso já tenha encontrado outra regra antes).
 * @return
 */
int retorna_regra(char *palavra_reversa, int num_regra) {

    int tamanho;

    //Laço que percorre o vetor de regras.
    for (int i = num_regra; i < QUANTIDADE_REGRAS; i++) {

        //Pega o sufixo da palavra passada como parametro no mesmo tamanho da regra encontrada acima.
        //É feito para poder comparar o sufixo da palavra reversa passada como parâmetro com a regra do vetor de regras.
        tamanho = strlen(regras[i].sufixo);

        //Compara o sufixo da palavra reversa com a regra. Se forem iguais retorna o número da regra no vetor de regras
        if (strncmp(palavra_reversa, regras[i].sufixo, tamanho) == 0) return i;

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
char * stemmer(char *palavra) {
	
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
		num_regra = retorna_regra(palavra_reversa, num_regra);

		//Se num_regra = -1 então não foi encontrada nenhuma regra para ser aplicada e o stem foi encontrado.
		if (num_regra == -1) {
			break;
		}

		//copia o sufixo para a variável regra.
		strcpy(regra, regras[num_regra].sufixo);

		//armazena o tamanho do sufixo.
		int tamanho_sufixo = strlen(regras[num_regra].sufixo);
		//armazena a quantidade de caracteres que deverão ser retirados do sufixo.
		int tamanho = tamanho_sufixo - regras[num_regra].qtde_retirada;


		//Os tr?s passos a seguir montam o stem na forma inversa.
		//Recebe os caracteres que serão acrescentados, caso tenha algum, senão será vazio.
		strcpy(stem_reverso, regras[num_regra].rep);
		//concatena parte da palavra reversa (apenas a quantidade de caracteres da variável tamanho) ao stem reverso.
		strncat(stem_reverso, palavra_reversa, tamanho);
		//concatena a palavra reversa a partir da posição tamanho sufixo até o final da palavra ao stem reverso.
		strcat(stem_reverso, &(palavra_reversa[tamanho_sufixo]));

		/*ex.: connected
		       detcennoc
		stem_reverso = "" porque não tem caracteres para serem acrescentados.
		stem_reverso = "" porque a variável tamanho é igual a 0 pois nesse caso é para retirar o sufixo ed completo.
		stem_reverso = "tcennoc" pois concatena a palavra reversa a partir da posição 2 (ou seja retira o sufixo)
		*/

		if (valida_stem(stem_reverso)) {
			strcpy(palavra_reversa, stem_reverso);
			//Se encontrou o sinal de finalização então o stem foi encontrado.
			if (regras[num_regra].final[0] == '.') break;
			//Se encontrou o sinal para continuar volta ao primeiro registro da tabela de regras.
			//Isso é feito para que possa ser aplicada uma nova regra do stemming.
			if (regras[num_regra].final[0] == '>') num_regra = 0;
		}
		else {
			//Adicona mais um para ir para próxima regra.
			break;
		}

		num_regra++;
	}

	//Chama rotina para colocar a palavra na ordem certa pois a mesma foi revertida anteriormente.
	reverso(palavra_reversa, palavra);
	//Retornar ponteiro
	return palavra;

}

/**
 * Converte string passada como parâmetro para minúsculo.
 * @param palavra string que será convertida
 */
void minusculo(char palavra[]) {
	
	for(int i = 0; palavra[i]; i++){
		if (isspace (palavra[i])) {
			palavra [i] = '\0';
		}
		else {
			palavra[i] = tolower(palavra[i]);
		}
	}
}

/**
 * Abre arquivo com os tokens para leitura.
 * @param filename nome do arquivo
 * @param imprime  indica se é para imprimir na tela as palavras com os stems (0 - não imprime; 1 - imprime).
 */
bool le_arquivo(char *filename, char *imprime)
{
	FILE *arquivo;
    FILE *arqsaida;
	char palavra[TAMANHO_PALAVRA];
	char palavra_aux[TAMANHO_PALAVRA];
	int i, result;


	// Abre um arquivo texto para leitura
	arquivo = fopen(filename, "r");
	
	// Se houve erro na abertura
	if (arquivo == NULL)
	{
		printf("Houve um problema na abertura do arquivo.\n");
		return false;
	}


	// Cria um arquivo texto para gravação
    arqsaida = fopen("saida.txt", "w");
	// Se não conseguiu criar
    if (arqsaida == NULL)
    {
        printf("Problemas na criacao do arquivo\n");
        return false;
    }
	
	i = 1;
	while (!feof(arquivo))
	{		
		// Lê uma linha até 255 caracteres (inclusive com o '\n') e armazena na variável palavra.
		if (fgets(palavra, TAMANHO_PALAVRA, arquivo)) { 

			//Converte a palavra para minúsculo.
			minusculo(palavra);
			//Faz uma cópia da variável palavra para palavra_aux
			strcpy(palavra_aux, palavra);
			stemmer(palavra_aux);

			if (imprime[0] == '1')
				printf("Linha => %d Palavra => %s Stem => %s\n\n", i, palavra, palavra_aux);

            result = fprintf(arqsaida,"Linha => %d Palavra => %s Stem => %s\n\n", i, palavra, palavra_aux);
            if (result == EOF)
                printf("Erro na gravacao\n");
		}

		i++;
	}
	fclose(arquivo);

    fclose(arqsaida);

	return true;
}


/**
 * Principal onde faz a chamada para ler o arquivo e processar o stemming.
 * @param argc
 * @param argv
 * @return
 */

int main(int argc, char *argv[])
{

	//Se não tiver o segundo argumento, pede para informar o nome do arquivo.
	if(argc != 3)
		return printf("%s\n", "Informe o nome do arquivo e a indicacao de imprimir ou nao na tela (0 - nao imprime; 1 - imprime).");

	struct timespec start, stop;
    double accum;
    clock_gettime( CLOCK_REALTIME, &start);
	
	le_arquivo(argv[1], argv[2]);
	
	clock_gettime( CLOCK_REALTIME, &stop);

    if ((stop.tv_nsec - start.tv_nsec) < 0)
        accum = (stop.tv_sec - start.tv_sec - 1) + (1000000000 + stop.tv_nsec - start.tv_nsec);
    else
        accum = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
	accum = accum / BILLION;
    printf( "%lf\n", accum );


	return 0;
}

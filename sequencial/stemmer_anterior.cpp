#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "rules.h"
#include <time.h>


//Retorna a regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada então retorna -1.
int retorna_regra(char *palavra, int num_regra) {

	int tamanho_sufixo;
	int tamanho_palavra;
	char sufixo_palavra[TAMANHO_PALAVRA];
		
	//Laço que percorre o vetor de regras.
	for (int i = num_regra; i < QUANTIDADE_REGRAS; i++) {
		// Pega a regra atual do vetor de regras.
		//strcpy(regra, regras[i].sufixo);
		
		//Pega o tamanho do sufixo do vetor de regras.
		//É feito para poder comparar o sufixo da palavra passada como parâmetro com a regra do vetor de regras.
		tamanho_sufixo = strlen(regras[i].sufixo);
		//
		tamanho_palavra = strlen(palavra);
		
		int tam = tamanho_palavra - tamanho_sufixo;
		
		strcpy(sufixo_palavra, &(palavra[tam]));
		printf("sufixo_palavra: %s \n", sufixo_palavra);		
		printf("palavra: %s sufixo: %s \n", palavra, regras[i].sufixo);		
		//Compara o sufixo da palavra com a regra. Se forem iguais retorna o número da regra no vetor de regras
		if (strncmp(sufixo_palavra, regras[i].sufixo, tamanho_sufixo) == 0) return i;

	}
	//Se não encontrou nenhuma regra que seja igual ao sufixo da palavra reversa então retorna -1.
	return -1;
}

//Verifica se a letra é vogal ou y.
bool letra_eh_vogal(char letra) {
	letra = tolower(letra);
	
	if (letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u' || letra == 'y') return true;
	return false;
}

//Verifica se a palavra tem vogal.
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


//Verifica se o stem é válido.
bool valida_stem(char * stem) {
	//Se o stem inicia com uma vogal então deve ter pelo menos duas letras.
	//ex.: owed ==> ow, owing ==> ow
	//mas não pode ser: ear ==> e
	
	if (letra_eh_vogal(stem[0]) == true) {
		
		if (strlen(stem) >= 2 )
			return true;
		else 
			return false;
	}
	else {
		// Se o stem inicia com uma consoante então deve ter pelo menos três letras.
		if (strlen(stem) < 3) return false;
		// e pelo menos uma das letras deve ser uma vogal ou y.
		// ex.: crying ==> cry e saying ==> say
		// mas não pode: string ==> str, meant ==> me, cement ==> ce
		if (string_tem_vogal(stem))
			return true;
		else
			return false;
	}

	return false;
}


//Retorna a string na forma inversa.
void reverso(char *palavra, char *palavra_reversa) {
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
char * stemmer(char *palavra) {
	int intacto = 1;
	
	//char palavra_reversa[TAMANHO_PALAVRA];
	int num_regra = 0;
	char regra[10];
	char vazia[] = "";
	char stem[TAMANHO_PALAVRA];

	//Chama rotina para colocar a palavra na ordem reversa.
	//reverso(palavra, palavra_reversa);
	
	//Laço que percorre as regras até encontrar um '.' ou 'end0' que indica que não tem mais regras para retirar sufixo.
	while (1) {
		//Chama rotina para tentar encontrar uma regra de remoção/substituição para a palavra.
		//A variável num_regra guarda o índice da última regra encontrada para que seja percorrido no vetor de regras a partir desse índice e não do índice zero do vetor.
		num_regra = retorna_regra(palavra, num_regra);
		printf("palavra: %s regra: %d \n", palavra, num_regra);

		//Se num_regra = -1 então não foi encontrada nenhuma regra para aplicar e o stem foi encontrado.
		if (num_regra == -1) {
			break;
		}
		//copia o sufixo para a variável regra.
		strcpy(regra, regras[num_regra].sufixo);
		
		//Se 
		if ((regras[num_regra].asterisco[0] != '*' || intacto) ) {
			
			//int tamanho_sufixo = strlen(regras[num_regra].sufixo);
			//int tamanho = tamanho_sufixo - regras[num_regra].qtde;								
			
			int tamanho = strlen(palavra) - regras[num_regra].qtde;
			printf("tamanho: %d \n", tamanho);
			strncat(stem, palavra, tamanho);
			printf("stem1: %s \n", stem);
			strcat(stem, regras[num_regra].rep);
			printf("stem2: %s \n", stem);
			
			//strcpy(stem, regras[num_regra].rep);
			//strncat(stem, palavra_reversa, tamanho);				
			//strcat(stem, &(palavra_reversa[tamanho_sufixo]));
			
			if (valida_stem(stem)) {							
				strcpy(palavra, stem);
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
	//reverso(palavra_reversa, palavra);
	//Retornar ponteiro
	return palavra;

} //end stemmer

//
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

void le_arquivo()
{
	char url[] = "arquivos/lista-palavras-ingles.txt";
	FILE *arquivo;
	char palavra[TAMANHO_PALAVRA];
	char stem[TAMANHO_PALAVRA];
	int i;


	// Abre um arquivo texto para leitura
	arquivo = fopen(url, "r");
	
	// Se houve erro na abertura
	if (arquivo == NULL)
	{
		printf("Houve um problema na abertura do arquivo.\n");
		return;
	}
	
	i = 1;
	while (!feof(arquivo))
	{		
		// Lê uma linha até 255 caracteres (inclusive com o '\n').
		if (fgets(palavra, TAMANHO_PALAVRA, arquivo)) { 
			//printf("linha %d : %s\n", i, palavra);
			minusculo(palavra);
			strcpy(stem, palavra);
			stemmer(stem);
			printf("palavra => %s stem => %s\n\n", palavra, stem);
		}
		//exit(0);
		i++;
	}
	fclose(arquivo);
}

int main(int argc, char *argv[])
{
/*	struct timespec start, stop;
    double accum;

    clock_gettime( CLOCK_REALTIME, &start);
	
	le_arquivo();
	
	clock_gettime( CLOCK_REALTIME, &stop);

	accum = ( stop.tv_sec - start.tv_sec ) + ( stop.tv_nsec - start.tv_nsec ) / BILLION;
    printf( "%lf\n", accum );
*/
	
	char palavra[] = "aardwolves";
	stemmer(palavra);
	printf("%s\n\n", palavra);
	return 0;
}

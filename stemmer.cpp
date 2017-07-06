#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "rules.h"



//Retorna o numero da primeira regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada então retorna -1.
int primeira_regra(char *palavra_reversa, int num_regra) {

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

//
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
bool valida_stem(char * stem_reverso) {
	//if (preg_match("/[aeiouy]$/", utf8_encode(stem_reverso))) {
	//Se o stem inicia com uma vogal então deve ter pelo menos duas letras. ("owed"/"owing" --> "ow", but not "ear" --> "e")
	//ex.: owed ==> ow, owing ==> ow
	//mas não pode: ear ==> e
	
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
	//boolean stem_found = False;
	char palavra_reversa[64];
	int num_regra = 0;
	char regra[10];
	char vazia[] = "";
	char stem_reverso[64];

	//Chama rotina para colocar a palavra na ordem reversa.
	reverso(palavra, palavra_reversa);
	
	//Laço que percorre as regras até encontrar um '.' ou 'end0' que indica que não tem mais regras para retirar sufixo.
	while (1) {
		//Chama rotina para tentar encontrar uma regra de remoção/substituição para a palavra.
		//A variável num_regra guarda o índice da última regra encontrada para que seja percorrido no vetor de regras a partir desse índice e não do índice zero do vetor.
		num_regra = primeira_regra(palavra_reversa, num_regra);

		//Se num_regra = -1 então não foi encontrada nenhuma regra para aplicar e o stem foi encontrado.
		if (num_regra == -1) {
			break;
		}
		//
		strcpy(regra, regras[num_regra].sufixo);
		
		//Se 
		if ((regras[num_regra].asterisco[0] != '*' || intacto) ) {
			
			int tamanho_sufixo = strlen(regras[num_regra].sufixo);
			int tamanho = tamanho_sufixo - regras[num_regra].qtde;								
			//printf("tamanho:%d\n\n", tamanho);			
			//printf("tamanho_sufixo:%d\n\n", tamanho_sufixo);	
			
			strcpy(stem_reverso, regras[num_regra].rep);
			//printf("1:%s\n\n", stem_reverso);
			strncat(stem_reverso, palavra_reversa, tamanho);				
			//printf("2:%s\n\n", stem_reverso);
			strcat(stem_reverso, &(palavra_reversa[tamanho_sufixo]));
			//printf("3:%s\n\n", stem_reverso);
			
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
	char linha[255];
	int i;

	// Abre um arquivo texto para leitura
	arquivo = fopen(url, "r");
	
	// Se houve erro na abertura
	if (arquivo == NULL)
	{
		printf("Houve um problema na abertura do arquivo.\n");
		return;
	}

	char palavra[255];
	i = 1;
	while (!feof(arquivo))
	{		
		// Lê uma linha até 255 caracteres (inclusive com o '\n').
		if (fgets(linha, 255, arquivo)) { 
			printf("linha %d : %s\n", i, linha);
			minusculo(linha);
			strcpy(palavra, linha);
			stemmer(palavra);
			printf("stem => %s\n\n", palavra);
		}
		//exit(0);
		i++;
	}
	fclose(arquivo);
}

int main(int argc, char *argv[])
{
	//le_arquivo();
	char palavra[] = "greatness";
	stemmer(palavra);
	printf("%s\n\n", palavra);
	return 0;
}

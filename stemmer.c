#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "sequencial/rules.h"
typedef enum { false, true } bool;

//Retorna o numero da primeira regra, do array de regras, que pode ser aplicada a uma palavra. Se nenhuma regra puder ser aplicada ent�o retorna -1.
int primeira_regra(char *palavra_reversa, int num_regra) {

	int tamanho;
	
	//La�o que percorre o vetor de regras.
	for (int i = num_regra; i < QUANTIDADE_REGRAS; i++) {
		// Pega a regra atual do vetor de regras.
		//strcpy(regra, regras[i].sufixo);

		//Pega o sufixo da palavra passada como parametro no mesmo tamanho da regra encontrada acima.
		//� feito para poder comparar o sufixo da palavra reversa passada como par�metro com a regra do vetor de regras.
		tamanho = strlen(regras[i].sufixo);
		//Compara o sufixo da palavra reversa com a regra. Se forem iguais retorna o n�mero da regra no vetor de regras
		if (strncmp(palavra_reversa, regras[i].sufixo, tamanho) == 0) return i;

	}
	//Se n�o encontrou nenhuma regra que seja igual ao sufixo da palavra reversa ent�o retorna -1.
	return -1;
}

//Verifica se a palavra tem vogal.
bool string_tem_vogal(char * palavra)
{
	int i, j;
	for (j = 0; j < strlen(palavra); i++) {
	{
		if (letra_eh_vogal(palavra[j])) {
				return true;
		}
	}
	return false;
}	

//
bool letra_eh_vogal(char letra) {
	letra = tolower(letra);
	if (letra == 'a' || letra == 'e' || letra == 'i' || letra == 'o' || letra == 'u' || letra == 'y') return true;
	return false;
}

//Verifica se o stem � v�lido.
bool valida_stem(char * stem_reverso) {
	//if (preg_match("/[aeiouy]$/", utf8_encode(stem_reverso))) {
	//Se o stem inicia com uma vogal ent�o deve ter pelo menos duas letras. ("owed"/"owing" --> "ow", but not "ear" --> "e")
	//ex.: owed ==> ow, owing ==> ow
	//mas n�o pode: ear ==> e
	if (letra_eh_vogal(stem_reverso[0]) == true) {
		if (strlen(stem_reverso) >= 2 )
			return true;
		else 
			return false;
	}
	else {
		// Se o stem inicia com uma consoante ent�o deve ter pelo menos tr�s letras.
		if (strlen(stem_reverso) < 3) return false;
		// e pelo menos uma das letras deve ser uma vogal ou y.
		// ex.: crying ==> cry e saying ==> say
		// mas n�o pode: string ==> str, meant ==> me, cement ==> ce
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
	
	k = strlen(palavra);
	
	for(j = 0; j <= k - 1; j++)
	{
		palavra_reversa[j] = palavra[k];
		k--;
	}

}

// retorna o stem (radical) para uma palavra.
char * stemmer(char *palavra) {
	int intact = 1;
	//boolean stem_found = False;

	char palavra_reversa[64];
	int num_regra = 0;
	char regra[10];
	char vazia[] = "";
	char stem_reverso[64];

	//Chama rotina para colocar a palavra na ordem reversa.
	reverso(palavra, palavra_reversa);
	//La�o que percorre as regras at� encontrar um '.' ou 'end0' que indica que n�o tem mais regras para retirar sufixo.
	while (1) {
		//Chama rotina para tentar encontrar uma regra de remo��o/substitui��o para a palavra.
		//A vari�vel num_regra guarda o �ndice da �ltima regra encontrada para que seja percorrido no vetor de regras a partir desse �ndice e n�o do �ndice zero do vetor.
		num_regra = primeira_regra(palavra_reversa, num_regra);

		//Se num_regra = -1 ent�o n�o foi encontrada nenhuma regra para aplicar e o stem foi encontrado.
		if (num_regra == -1) {
			break;
		}
		//
		strcpy(regra, regras[num_regra].sufixo);
		
		//Se 
		if ((regras[num_regra].asterisco != '*') ) {
			
			int tamanho_sufixo = strlen(regras[num_regra].sufixo);
			int tamanho = tamanho_sufixo - regras[num_regra].qtde;								
			
			strcpy(stem_reverso, regras[num_regra].rep);
			strncat(stem_reverso, palavra_reversa, tamanho);				
			strcat(stem_reverso, &(palavra_reversa[tamanho_sufixo]));
			
			if (valida_stem(stem_reverso)) {
				strcpy(palavra_reversa, stem_reverso);
				if (regras[num_regra].final == '.') break;
			}
			else {
				//Adicona mais um para ir para outra regra.
				num_regra++;
			}
		}
		else {
			//Adicona mais um para ir para outra regra.
			num_regra++;
		}
	}

	//Chama rotina para colocar a palavra na ordem certa pois a mesma foi revertida anteriormente.
	reverso(palavra_reversa, palavra);
	//Retornar ponteiro
	return palavra;

} //end stemmer

int main(int argc, char *argv[])
{
	stemmer("connecting");
	return 0;
}


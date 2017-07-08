#include <stdint.h>
#include <stddef.h>
#include <host_defines.h>




/**
 * Determina a quantidade de bytes que compõem uma string codificada em ASCII.
 *
 * @param s String codificada em UTF-8.
 * @return
 */
__device__
        size_t d_strlen (char* s)
{
    char* nav = s;

    while (*nav)
    {
        ++nav;
    }

    return (nav - s);
}



/**
 * Copia 'n' caracteres da string 'from' para o buffer apontado por 'to'.
 *
 * @param to Ponteiro para a região da memória que receberá a string.
 * @param from Ponteiro para a string que será copiada.
 * @param n Número máximo de caracteres que deve ser copiado.
 * @return
 */
__device__
char* d_strncpy (char* to, const char* from, size_t n)
{
    char* s = to;

    while ((n > 0) && (*from != '\0'))
    {
        *s++ = *from++;
        --n;
    }

    while (n > 0)
    {
        *s++ = '\0';
        --n;
    }

    return to;
}



/**
 * Determina se um caractere qualquer é uma letra ou não.
 *
 * Obs.: implementação para a GPU da função 'isalpha' disponível na STDLIB, mas indisponível no CUDART.
 *
 * @param c Caractere que será analisado.
 * @return
 */
__device__
bool d_isalpha (char c)
{
    return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
}



/**
 * Determina se um caractere qualquer é um dígito ou não.
 *
 * Obs.: implementação para a GPU da função 'isdigit' disponível na STDLIB, mas indisponível no CUDART.
 *
 * @param c Caractere que será analisado.
 * @return
 */
__device__
bool d_isdigit (char c)
{
    return (c >= '0' && c <= '9');
}



/**
 * Converte um caractere para seu equivalente em minúsculo.
 *
 * @param c Caractere que será analisado.
 * @return Letra equivalente em minúsculo caso seja uma letra (A-Z), ou o próprio char caso não seja uma letra.
 */
__device__
char d_tolower (char c)
{
    if ((c >= 'A') && (c <= 'Z'))
    {
        return (char) (c + 32);
    }
    else
    {
        return c;
    }
}


/**
 * Procura uma substring dentro de uma string qualquer.
 *
 * @param string String que será usada para a busca.
 * @param substring Substring que deve ser localizada.
 * @return
 */
__device__
char* d_strstr (const char* string, const char* substring)
{
    char* nav_str, * nav_sub;
    //char c;

    nav_sub = (char*) substring;

    if (*nav_sub == '\0')
    {
        return (char*) string;
    }

    while (*string != '\0')
    {
        //
        // Procura pela coincidência do primeiro caractere.
        //
        if (*string++ != *nav_sub)
        {
            continue;
        }

        nav_str = (char*) string;

        while (true)
        {
            if (*nav_sub == '\0')
            {
                //
                // Localizamos a substring dentro da string. Retorna o valor atual do ponteiro.
                //
                return (char *) string;
            }
            else if (*nav_str++ != *nav_sub++)
            {
                break;
            }
        }

        //
        // Restaura o ponteiro para continuarmos procurando...
        //
        nav_sub = (char*) substring;
    }


    //return nullptr;
    return 0;
}

/**
 * Compara 'n' caracteres das duas strings passadas como parâmetro.
 *
 * @param string Ponteiro para a região da memória da primeira string.
 * @param string1 Ponteiro para a segunda string que será comparada.
 * @param n Número máximo de caracteres que deve ser comparado.
 * @return
 */
__device__
bool d_strncmp (char* string, const char* string1, size_t n)
{

    while ((n > 0) && (*string != '\0'))
    {
        if (string[n] != string1[n])
            return false;
        --n;
    }

    return true;
}
/*int strncmp(const char* s1, const char* s2, size_t n)
{
    while(n--)
        if(*s1++!=*s2++)
            return *(unsigned char*)(s1 - 1) - *(unsigned char*)(s2 - 1);
    return 0;
}*/

/**
 * Concatena uma string em outra de acordo com o tamanho
 *
 * @param to Ponteiro para a região da memória que receberá a string.
 * @param from Ponteiro para a string que será copiada.
 * @param n Número máximo de caracteres que deve ser copiado.
 * @return
 */
__device__
char *d_strncat(char *to, const char *from, size_t n)
{
    char *ret = to;
    while (*to)
        to++;
    while (n--)
        if (!(*to++ = *from++))
            return ret;
    *to = 0;
    return ret;
}
/*
char *d_strncat(char *dest, const char *src, size_t n)
{
    char *ret = dest;
    while (*dest)
        dest++;
    while (n--)
        if (!(*dest++ = *src++))
            return ret;
    *dest = 0;
    return ret;
}*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

bool d_strncmp(char *string, char *string1, size_t n) {

    while ((n > 0) && (*string != '\0')) {
        printf("pri=%s  seg=%s\n\n", &string[n], &string1[n]);
        if (string[n] != string1[n])
            return false;
        --n;
    }

    return true;
}

/*char* d_strncat (char* to, const char* from, size_t n)
{
    char* s = to;

    while ((n > 0) && (*from != '\0'))
    {
        s[strlen(s)]++ = from[n]++;
        --n;
    }

    while (n > 0)
    {
        *s++ = '\0';
        --n;
    }

    return to;
}*/

char *custom_strncat(char *s, const char *t, size_t n) {
    size_t slen = strlen(s);
    char *pend = s + slen;

    //if(slen + n >= bfsize)
    //    return NULL;

    while(n--)
        *pend++ = *t++;

    return s;
}

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
}


int main(int argc, char *argv[]) {


    char * string1 = argv[1];
    char * string2 = argv[2];
    //printf("pri=%s  seg=%s", string1, string2);
/*    if (d_strncmp(string1, string2, strlen(string1)))
        printf("%s\n", "iguais");
    else
        printf("%s\n", "dif");*/
    char * retorno = d_strncat(string1, string2, strlen(string2));
    printf("%s\n", retorno);
}
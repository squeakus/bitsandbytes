#include <stdio.h>

#include "util.h"

char* itoa(int value, char* str, int radix) {
    static char dig[] =
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz";
    int n = 0, neg = 0;
    unsigned int v;
    char*p, *q;
    char c;
 
    if (radix == 10 && value < 0) {
        value = -value;
        neg = 1;
    }
    v = value;
    do {
        str[n++] = dig[v%radix];
        v /= radix;
    } while (v);

    if (radix == 2)
      while (n < 32)
        str[n++] = dig[0];

    if (neg)
        str[n++] = '-';
    str[n] = '\0';

    for (p = str, q = p + n-1; p < q; ++p, --q) {
      c = *p;
      *p = *q;
      *q = c;
    }

    return str;
}


#include<stdio.h>

typedef enum { false, true } bool;

void binaryval(unsigned char c)
{
    printf("idx: 76543210\n");
    printf("val: ");
    for (int i = 0; i < 8; i++) {
      printf("%d", !!((c << i) & 0x80));
    }
    printf("\n");
}


int main()
{
    unsigned char a, opt;
    int b, bit;
    bool run = true;
    printf("Please enter a value:");
    scanf("%hhd", &a);
    binaryval(a);

    while(run){
      printf("Would you like to (s)et, (c)lear, (f)lip, (r)ead or (q)uit?\n");
      scanf(" %c", &opt);
      if (opt =='q'){
        printf("quitting");
        break;
      }
      else if (opt =='s'){
        // bitwise OR
        printf("which bit you would like to set:");
        scanf("%d", &b);
        a |= 1 << b;
      }
      else if (opt =='c'){
        // bitwise AND
        printf("which bit you would like to clear:");
        scanf("%d", &b);
        a &= ~(1 << b);
      }
      else if (opt =='f'){
        //XOR
        printf("which bit you would like to flip:");
        scanf("%d", &b);
        a ^= 1 << b;
      }
      else if (opt =='r'){
        printf("which bit you would like to read:");
        scanf("%d", &b);
        bit = (a >> b) & 1;
        printf("bitvalue: %d\n", bit);
      }

      printf("after set:\n");
      binaryval(a);
    }
      return 0;
}

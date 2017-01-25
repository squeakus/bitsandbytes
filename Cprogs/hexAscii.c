#include <stdio.h>
#include <stdlib.h>

char hexToAscii(char first, char second)
{
  char hex[5], *stop;
  hex[0] = '0';
  hex[1] = 'x';
  hex[2] = first;
  hex[3] = second;
  hex[4] = 0;
  return strtol(hex, &stop, 16);
}

int isValid(char test)
{
  int result =0;
  switch(test)
    {
       case'0':
       case'1':
       case'2':
       case'3':
       case'4':
       case'5':
       case'6':
       case'7':
       case'8':
       case'9':
       case'a':
       case'b':
       case'c':
       case'd':
       case'e':
       case'f':
	 result =1;
	 break;
    }
  return result;
}

int main(int argc, char* argv[])
{
  char input;
  char a;
  char b;
  while (((input = getchar()) != EOF))
    {
      if(isValid(input)&& !(isValid(a)))
	{
	  a = input;
	}
      else if (isValid(input)&& !(isValid(b)))
	{
	  b = input;
	}

      if(isValid(a)&&isValid(b))
	{
	       
	  putchar(hexToAscii(a,b));
	  a = b =0;
	}
    }
}

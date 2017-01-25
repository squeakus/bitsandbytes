#include<stdio.h>
#include <stdlib.h>
#include <string.h>



char* StrReverse(char* str)
{
  char *temp, *ptr;
	int len, i;

	temp=str;
	for(len=0; *temp !='\0';temp++, len++);
	
	ptr=malloc(sizeof(char)*(len+1));
	
	for(i=len-1; i>=0; i--) 
		ptr[len-i-1]=str[i];
	
	ptr[len]='\0';
	return ptr;
}


int main(int argc, char** argv) {
  char str[80];    
//  fscanf(stdin,"%[^\n]",str);  
  while(fscanf(stdin, "%s", str) == 2)
    {
     fprintf(stdout,"%s \n",str);
    }
fprintf(stdout,"%s",StrReverse(str));

//  int i;
// for (i = 1 ; i < argc; ++i)
// { 
//   fprintf(stdout,"%s ", StrReverse(argv[i]));
// }
return 0;
}




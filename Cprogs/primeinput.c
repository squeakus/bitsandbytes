#include <stdio.h>

int main()
{
  int flag, i, j;
  int begin,end;

  //reading user input
  printf("\nplease enter start value: ");
  scanf("%d",&begin);
  printf("\nplease enter end value: ");
  scanf("%d",&end);
  printf("The range you have chosen is: %d %d",begin,end);
  //TODO: verify input, maybe have default values


  // this method uses a variation of Eratosthenes Sieve to calculate primes
  for(i=begin;i<=end;i++)
    {
      flag=0;
      //if nothing divides in twice theres no point in checking further
      for(j=2;j<=i/2;j++)
	{
	  if(i%j==0)
	    {
	      flag=1;
	      break;
	    }
	}
      if(flag==0)
	printf("\n%d", i);
    }
}

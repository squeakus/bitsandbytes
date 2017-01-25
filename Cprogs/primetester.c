#include <stdio.h>

int main()
{
  int flag, i, j;

  // this method uses a variation of Eratosthenes Sieve to calculate primes
  for(i=1000;i<=1100;i++)
    {
      flag=0;
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

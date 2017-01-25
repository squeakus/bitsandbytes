/*
    This subroutine multiplies the matrices A(Transpose)*B 

		A(p,n) B(p,m)

		Updated 2/1/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int matXT(double *C,double *A,double *B, int n,int m,int p)
{
	int i,j,k;

	memset(C,0,n*m*sizeof(double));
	for( i = 0; i < n; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
	   	for( k = 0; k < p; ++k )
		{
		   *(C+m*i+j) += (*(A+n*k+i))*(*(B+m*k+j));
		}
		/*printf("%d %d %10.7f ",i,j,*(C+m*i+j));*/
	   }
	   /*printf("\n");*/
	}
	return 1;
}


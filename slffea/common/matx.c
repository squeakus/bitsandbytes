/*
    This subroutine multiplies the matrices A*B 

                A(n,p) B(p,m)

	Updated 2/1/00

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int matX(double *C, double *A, double *B, int n, int m, int p)
{
	int i, j, k;
	memset(C, 0, n*m*sizeof(double));
	for( i = 0; i < n; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		for( k = 0; k < p; ++k )
		{
		   *(C+m*i+j) += (*(A+p*i+k))*(*(B+m*k+j));
		}
		/*printf("%d %d %10.7f ", i, j, *(C+m*i+j));*/
	   }
	   /*printf("\n");*/
	}
	return 1;
}


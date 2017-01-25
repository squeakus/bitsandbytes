/*
    This subroutine creates the dyadic product of
    2 vectors A and B 

                    A(p) B(p)

              Updated 11/7/00

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

int dyadicX(double *C,double *A,double *B, int p)
{
	int i,j,k;
	memset(C,0,p*p*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < p; ++j )
	   {
		*(C+p*i+j) = (*(A+i))*(*(B+j));
		/*printf("%d %d %10.7f ",i,j,*(C+p*i+j));*/
	   }
	   /*printf("\n");*/
	}
	return 1;
}


/*
     This library function assembles the B, B_T, B_T2 matrices
     in [Btrans][D][B] for thermal brick elements.

     It is based on the subroutine QDCB from the book
     "The Finite Element Method" by Thomas Hughes, page 780.

     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999, 2000, 2001, 2002  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../brick/brconst.h"

int brickB_T(double *shg, double *B_T)
{
/*
 ....  SET UP THE TEMPERATURE GRADIENT MATRIX "B" FOR
       THREE-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX

		Updated 4/7/00
*/
	int i;
	for( i = 0; i < npel; ++i )
	{
		*(B_T+i)          = *(shg+i);
		*(B_T+Tneqel*1+i) = *(shg+npel*1+i);
		*(B_T+Tneqel*2+i) = *(shg+npel*2+i); 
	}
	return 1;
}

int brickB_T2(double *shg, double *B_T2)
{
/*
     This library function assembles the B_T2 matrix in
       ..
     [ q ][B_T2] and [T][B_T2] for brick elements.

		Updated 4/7/00
*/

	int i;
	for( i = 0; i < npel; ++i )
	{
		*(B_T2+i) = *(shg+i);
	}
	return 1;
}

int brickB_TB(double *shl, double *B_TB)
{
/*
     This library function assembles the B_TB matrix in
     [TB][B_TB] for brick elements.

		Updated 4/11/00
*/

	int i;
	for( i = 0; i < npel_film; ++i )
	{
		*(B_TB+i) = *(shl+i);
	}
	return 1;
}


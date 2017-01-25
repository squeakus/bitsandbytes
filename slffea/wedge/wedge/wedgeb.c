/*
     This library function assembles the B matrix in [Btrans][D][B]
     for wedge elements.

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
#include "weconst.h"

int wedgeB(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       THREE-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX

		Updated 4/7/00
*/
	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel; ++i )
	{
		i2      =ndof*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B+i2m2)         = *(shg+i);
		*(B+neqel*1+i2m1) = *(shg+npel*1+i);
		*(B+neqel*2+i2)   = *(shg+npel*2+i); 
		*(B+neqel*3+i2m2) = *(shg+npel*1+i);
		*(B+neqel*3+i2m1) = *(shg+i);
		*(B+neqel*4+i2m2) = *(shg+npel*2+i);
		*(B+neqel*4+i2)   = *(shg+i);
		*(B+neqel*5+i2m1) = *(shg+npel*2+i);
		*(B+neqel*5+i2)   = *(shg+npel*1+i);

	}
	return 1;
}

int wedgeB_mass(double *shg, double *B_mass)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for wedge elements.

		Updated 4/7/00
*/

	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel; ++i )
	{
		i2      =ndof*i+2;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B_mass+i2m2)         = *(shg+i);
		*(B_mass+neqel*1+i2m1) = *(shg+i);
		*(B_mass+neqel*2+i2)   = *(shg+i); 
	}
	return 1;
}

int wedgeBomega(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       THREE-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE OMEGA MATRIX

		Updated 4/7/00
*/
	int i,i2,i2m1,i2m2;
	for( i = 1; i < 9; ++i )
	{
		i2      =ndof*i-1;
		i2m1    =i2-1;
		i2m2    =i2-2;

		*(B+neqel*3+i2m2) = -*(shg+2+i);
		*(B+neqel*3+i2m1) = *(shg+1+i);
		*(B+neqel*4+i2m2) = *(shg+3+i);
		*(B+neqel*4+i2)   = -*(shg+1+i);
		*(B+neqel*5+i2m1) = -*(shg+3+i);
		*(B+neqel*5+i2)   = *(shg+2+i);
	}
	return 1;
}

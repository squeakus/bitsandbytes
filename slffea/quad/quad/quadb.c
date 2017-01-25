/*
     This library function assembles the B matrix in [Btrans][D][B]
     for quad elements.

     It is a C translation of the subroutine QDCB from the book
     "The Finite Element Method" by Thomas Hughes, page 780.

     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qdconst.h"

int quadB(double *shg, double *B)
{
/*
 ....  SET UP THE STRAIN-DISPLACEMENT MATRIX "B" FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS
       FOR THE D MATRIX
		Updated 8/30/06
*/
	int i,i2,i2m1;
	for( i = 0; i < npel; ++i )
	{
		i2      =ndof2*i+1;
		i2m1    =i2-1;

		*(B+i2m1)         = *(shg+i);
             /* *(B+i2)           = 0.0;
		*(B+neqel8*1+i2m1) = 0.0; */
		*(B+neqel8*1+i2)   = *(shg+npel*1+i); 
		*(B+neqel8*2+i2m1) = *(shg+npel*1+i);
		*(B+neqel8*2+i2)   = *(shg+i);
	}
        return 1;
}

int quadB_mass(double *shg, double *B_mass)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for quad elements.

                Updated 8/30/06
*/

	int i,i2,i2m1,i2m2;
	for( i = 0; i < npel; ++i )
	{
		i2      =ndof2*i+1;
		i2m1    =i2-1;

		*(B_mass+i2m1)       = *(shg+i);
		*(B_mass+neqel8*1+i2) = *(shg+i);
	}
	return 1;
}

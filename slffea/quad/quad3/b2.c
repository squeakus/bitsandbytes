/*
     This library function assembles the [B] matrix in:

            [Btrans][B]

     the Bedge] matrix in:

            [Bedge trans][Bedge]:

     the [curlBedge] matrix in:

            [curlBedge trans][curlBedge]:

     and the [gradB] matrix in:

            [gradB trans][Bedge], [Bedge trans][gradB], [gradB trans][gradB]

     for quad elements.  It is for eletromagnetism.  All these 
     matrices are in 8.41 to 8.45 on page 246 of "The Finite Element
     Method in Electromagnetics" by Jianming Jin.

     It is a C translation of the subroutine QDCB from the book
     "The Finite Element Method" by Thomas Hughes, page 780.

     SLFFEA source file
     Version:  1.1
     Copyright (C) 1999, 2000  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qd3const.h"

int quad3B(double *dcdx, double *shl, double *shg, double *B,
	double *gradB, double *Bedge, double *curlBedge,
	double *length)
{
/*
 ....  SET UP THE SHAPE FUNCTION MATRICES FOR
       TWO-DIMENSIONAL CONTINUUM ELEMENTS

		Updated 12/11/00
*/
	int i,i2,i2m1;
	for( i = 0; i < EMneqel; ++i )
	{
                *(B+i)             = *(shg+npel*2+i);

        	*(gradB+i)         = *(shg+i);
        	*(gradB+EMneqel*1+i) = *(shg+npel*1+i); 
	}

/* The shl used to form Bedge comes from the shape function derivatives
   with respect to xi and eta.  I choose to do it this way because
   of expediancy, although it is hard to follow.  What you need to do 
   is compare the shape function derivatives for N with eq. 8.22-2.25 in 
   Jin's book. 
*/

#if 1
        *(Bedge)           = *(dcdx)*(*(shl + 1))*(*(length));
        *(Bedge + 1)       = *(dcdx)*(*(shl + 2))*(*(length + 2));
        *(Bedge + 2)       = *(dcdx+2)*(*(shl + npel*1 + 3))*(*(length + 1));
        *(Bedge + 3)       = *(dcdx+2)*(*(shl + npel*1 + 2))*(*(length + 3));
        *(Bedge+EMneqel*1)   = *(dcdx+1)*(*(shl + 1))*(*(length));
        *(Bedge+EMneqel*1+1) = *(dcdx+1)*(*(shl + 2))*(*(length + 2));
        *(Bedge+EMneqel*1+2) = *(dcdx+3)*(*(shl + npel*1 + 3))*(*(length + 1));
        *(Bedge+EMneqel*1+3) = *(dcdx+3)*(*(shl + npel*1 + 2))*(*(length + 3));

        *(curlBedge)     =
		*(length)*(*(dcdx)*(*(dcdx+3))-*(dcdx+1)*(*(dcdx+2)))/4.0;
        *(curlBedge + 1) =
		*(length + 2)*(*(dcdx+2)*(*(dcdx+1))-*(dcdx+3)*(*(dcdx)))/4.0;
        *(curlBedge + 2) =
		*(length + 1)*(*(dcdx+2)*(*(dcdx+1))-*(dcdx+3)*(*(dcdx)))/4.0;
        *(curlBedge + 3) =
		*(length + 3)*(*(dcdx)*(*(dcdx+3))-*(dcdx+1)*(*(dcdx+2)))/4.0;
#endif

#if 0
        *(Bedge)           = *(dcdx)*(*(shl + 1))*(*(length));
        *(Bedge + 1)       = *(dcdx+2)*(*(shl + npel*1 + 3))*(*(length + 1));
        *(Bedge + 2)       = *(dcdx)*(*(shl + 2))*(*(length + 2));
        *(Bedge + 3)       = *(dcdx+2)*(*(shl + npel*1 + 2))*(*(length + 3));
        *(Bedge+EMneqel*1)   = *(dcdx+1)*(*(shl + 1))*(*(length));
        *(Bedge+EMneqel*1+1) = *(dcdx+3)*(*(shl + npel*1 + 3))*(*(length + 1));
        *(Bedge+EMneqel*1+2) = *(dcdx+1)*(*(shl + 2))*(*(length + 2));
        *(Bedge+EMneqel*1+3) = *(dcdx+3)*(*(shl + npel*1 + 2))*(*(length + 3));

        *(curlBedge)     =
		*(length)*(*(dcdx)*(*(dcdx+3))-*(dcdx+1)*(*(dcdx+2)))/4.0;
        *(curlBedge + 1) =
		*(length + 1)*(*(dcdx+2)*(*(dcdx+1))-*(dcdx+3)*(*(dcdx)))/4.0;
        *(curlBedge + 2) =
		*(length + 2)*(*(dcdx+2)*(*(dcdx+1))-*(dcdx+3)*(*(dcdx)))/4.0;
        *(curlBedge + 3) =
		*(length + 3)*(*(dcdx)*(*(dcdx+3))-*(dcdx+1)*(*(dcdx+2)))/4.0;
#endif

        return 1;
}

int quadB_mass(double *shg,double *B_mass)
{
/*
     This library function assembles the B_mass matrix in
     [B_mass trans][B_mass] for quad elements.

                Updated 8/17/00
*/

        int i,i2,i2m1,i2m2;
        for( i = 0; i < npel; ++i )
        {
                i2      =ndof*i+1;
                i2m1    =i2-1;

                *(B_mass+i2m1)       = *(shg+i);
                *(B_mass+neqel*1+i2) = *(shg+i);
        }
        return 1;
}

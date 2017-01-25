/*
     SLFFEA source file
     Version:  1.5
     Copyright (C) 1999, 2000, 2001, 2002  San Le

     The source code contained in this file is released under the
     terms of the GNU Library General Public License.
*/

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../brick/brconst.h"

int dotX(double *,double *, double *, int );

int brshl_film( double g, double *shl_film)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a 4 node face element at the gauss points.

     It is a C translation of the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

                        Updated 4/12/00
*/

	double ra[]={-0.50, 0.50, 0.50,-0.50};
	double sa[]={-0.50,-0.50, 0.50, 0.50};
	double temps,tempr,r,s;
	int i,j,k;

	for( k = 0; k < num_int_film; ++k )
	{
/* calculating the local dN/ds,dr matrix */

		r=g*(*(ra+k));
		s=g*(*(sa+k));
		for( i = 0; i < npel_film; ++i )
		{
		   tempr = pt5 + *(ra+i)*r;
		   temps = pt5 + *(sa+i)*s;
		   *(shl_film+npel_film*nsd*k+i) =
			*(ra+i)*temps;
		   *(shl_film+npel_film*nsd*k+npel_film*1+i) =
			tempr*(*(sa+i));
		   *(shl_film+npel_film*nsd*k+npel_film*2+i) =
			tempr*temps;
		}
		/*printf("\n");*/
	}
	return 1;
}

int brshface( double *dArea, int el, double *shl_film, double *xl)
{
/*
     This subroutine calculates the derivatives of the global coordinates
     x, y, z with respect to the local variables XI and ETA for
     a thermal brick element at the gauss points for one face of a 3-D element.
     Unlike brshg, we do not have to calculate the shape function derivatives
     with respect to the global coordinates x, y, z which are only needed for
     strains and stresses.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

     Also, the dArea term is based on the calculations given in
     "Vector Calculus", 3rd ed.  by Jarrold E. Marsden and
     Anthony J. Tromba, 1988, W. H. Freeman and Co., page 450.
     
     

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(dArea+k)  = differential element Area calculated at gauss point
       *(shl_film+npel_film*nsd*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl_film+npel_film*nsd*k+npel_film*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl_film+npel_film*nsd*k+npel_film*2+i) = LOCAL SHAPE FUNCTION
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int_film    = NUMBER OF INTEGRATION POINTS FOR FILM SURFACE, 4

                        Updated 9/25/01
*/

	double xs[6];
	double fdum, fdum2, fdum3;
	int check,i,j,k;

	for( k = 0; k < num_int_film; ++k )
	{

/* The jacobian, dx/dc, is calculated below */

	   for( j = 0; j < nsd - 1; ++j )
	   {
		for( i = 0; i < nsd; ++i )
		{
		   check=dotX((xs+(nsd-1)*i+j),(shl_film+npel_film*nsd*k+npel_film*j),
			(xl+npel_film*i),npel_film);
		}
	   }
	   fdum = *(xs)*(*(xs+3))-*(xs+1)*(*(xs+2));
	   fdum2 = *(xs+2)*(*(xs+5))-*(xs+3)*(*(xs+4));
	   fdum3 = *(xs)*(*(xs+5))-*(xs+1)*(*(xs+4));

	   *(dArea + k) = fdum*fdum + fdum2*fdum2 + fdum3*fdum3;
	   *(dArea + k) = sqrt(*(dArea + k));

	   /*printf("k %3d %14.5f %14.5f %14.5f %14.5f\n",k,fdum, fdum2, fdum3);*/

	}
	return 1; 
}


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
#include "qdconst.h"

int dotX(double *,double *, double *, int );

int qdshl( double g, double *shl, double *w)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a quad element at the gauss points.

     It is a C translation of the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

                        Updated 8/7/06
*/

	double ra[]={-0.50, 0.50, 0.50,-0.50};
	double sa[]={-0.50,-0.50, 0.50, 0.50};
	double temps,tempr,r,s;
	int i,j,k;

	for( k = 0; k < num_int; ++k )
	{
/* calculating the weights and local dN/ds,dr matrix */

		*(w+k)=1.0;

		r=g*(*(ra+k));
		s=g*(*(sa+k));
		for( i = 0; i < npel; ++i )
		{
		   tempr = pt5 + *(ra+i)*r;
		   temps = pt5 + *(sa+i)*s;
		   *(shl+npel*(nsd2+1)*k+i)=*(ra+i)*temps;
		   *(shl+npel*(nsd2+1)*k+npel*1+i)=tempr*(*(sa+i));
		   *(shl+npel*(nsd2+1)*k+npel*2+i)=tempr*temps;
		}
		/*printf("\n");*/
	}
	return 1;
}

int qdshl_node2(double *shl_node2)
{
/* 
   This subroutine is a condensed version of the shape function shl_node.  I have
   removed all of the zero terms which made up 75% of all the values in shl_node.

                        Updated 4/16/99
*/

/* For node 0 */
	*(shl_node2) =   -0.50; *(shl_node2+1) =  0.50; *(shl_node2+2) = -0.50;
	*(shl_node2+3) =  0.50;

/* For node 1 */
	*(shl_node2+4) = -0.50; *(shl_node2+5)  =  0.50; *(shl_node2+6)  = -0.50;
	*(shl_node2+7) =  0.50;

/* For node 2 */
	*(shl_node2+8) =  0.50; *(shl_node2+9) = -0.50; *(shl_node2+10) = -0.50;
	*(shl_node2+11) =  0.50;

/* For node 3 */
	*(shl_node2+12) =  0.50; *(shl_node2+13) = -0.50; *(shl_node2+14) = -0.50;
	*(shl_node2+15) =  0.50;

	return 1;
}



int qdshg( double *det, int el, double *shl, double *shg, double *xl)
{
/*
     This subroutine calculates the global shape function derivatives for
     a quad element at the gauss points.

     It is a C translation of the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR A FOUR-NODE QUADRALATERAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES(I LOOPS OVER X,Y)
       *(det+k)  = JACOBIAN DETERMINANT
       *(shl+npel*(nsd2+1)*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd2+1)*k+npel*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd2+1)*k+npel*2+i) = LOCAL SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+i) = X-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*1+i) = Y-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*2+i) = shl(npel*(nsd2+1)*k+npel*2+i)
       *(xs+2*i+j) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.4 

                        Updated 8/7/06
*/

	double xs[4],temp;
	int check,i,j,k;

	memcpy(shg,shl,sosh*sizeof(double));

	for( k = 0; k < num_int; ++k )
	{

/* The jacobian dx/dc is calculated below */

	   for( j = 0; j < nsd2; ++j )
	   {
		for( i = 0; i < nsd2; ++i )
		{
		   check=dotX((xs+nsd2*i+j),(shg+npel*(nsd2+1)*k+npel*j),
			(xl+npel*i),npel);
		}
	   }

	   *(det+k)=*(xs)*(*(xs+3))-*(xs+2)*(*(xs+1));
	   /*printf(" %9.5f %9.5f\n %9.5f %9.5f\n",
		*(xs),*(xs+1),*(xs+nsd2*1),*(xs+nsd2*1+1));
	   printf("%d %f\n", k, *(det+k));*/

	   if(*(det+k) <= 0.0 ) 
	   {
		printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+k),k);
		return 0;
	   }

/* The inverse of the jacobian, dc/dx, is calculated below */

	   temp=*(xs);
	   *(xs)=*(xs+3)/(*(det+k));
	   *(xs+1)*=-1./(*(det+k));
	   *(xs+2)*=-1./(*(det+k));
	   *(xs+3)=temp/(*(det+k));

	   for( i = 0; i < npel; ++i )
	   {
		*(shg+npel*(nsd2+1)*k+i) = *(xs)*(*(shl+npel*(nsd2+1)*k+i))+
			*(xs+2)*(*(shl+npel*(nsd2+1)*k+npel*1+i));
		*(shg+npel*(nsd2+1)*k+npel*1+i)=*(xs+1)*(*(shl+npel*(nsd2+1)*k+i))+
			*(xs+3)*(*(shl+npel*(nsd2+1)*k+npel*1+i));
		/*printf("%d %f %f %f\n", i, *(shg+npel*(nsd2+1)*k+i),
			*(shg+npel*(nsd2+1)*k+npel*1+i), 
			*(shg+npel*(nsd2+1)*k+npel*2+i));*/
	   }
	}
	return 1; 
}

int qdshg_mass( double *det, int el, double *shg, double *xl)
{
/*
     This subroutine calculates the determinant for
     a quad element at the gauss points for the calculation of
     the global mass matrix.  Unlike qdshg, we do not have to
     calculate the shape function derivatives with respect to
     the global coordinates x, y, z which are only needed for
     strains and stresses.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det+k)  = JACOBIAN DETERMINANT
       *(shg+npel*(nsd2+1)*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*2+i) = shl(npel*(nsd2+1)*k+npel*2+i)
       *(xs+nsd2*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 4

                        Updated 8/7/06
*/

	double xs[4],temp;
	int check,i,j,k;

	for( k = 0; k < num_int; ++k )
	{

/* The jacobian dx/dc is calculated below */

	   for( j = 0; j < nsd2; ++j )
	   {
		for( i = 0; i < nsd2; ++i )
		{
		   check=dotX((xs+nsd2*i+j),(shg+npel*(nsd2+1)*k+npel*j),
			(xl+npel*i),npel);
		}
	   }

	   *(det+k)=*(xs)*(*(xs+3))-*(xs+2)*(*(xs+1));
	   /*printf(" %9.5f %9.5f\n %9.5f %9.5f\n",
		*(xs),*(xs+1),*(xs+nsd2*1),*(xs+nsd2*1+1));
	   printf("%d %f\n", k, *(det+k));*/

	   if(*(det+k) <= 0.0 ) 
	   {
		printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+k),k);
		return 0;
	   }

	}
	return 1; 
}


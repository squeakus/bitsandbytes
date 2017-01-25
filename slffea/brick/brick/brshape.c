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
#include "brconst.h"

int dotX(double *,double *, double *, int );

int brshl( double g, double *shl, double *w)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a brick element at the gauss points.

     It is based on the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

 ....  CALCULATE INTEGRATION-RULE WEIGHTS, SHAPE FUNCTIONS AND
       LOCAL DERIVATIVES FOR A EIGHT-NODE HEXAHEDRAL ELEMENT
       r, s, t = LOCAL ELEMENT COORDINATES ("XI","ETA","ZETA", RESP.)
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(w+k)    = INTEGRATION-RULE WEIGHT
       i       = LOCAL NODE NUMBER
       k       = INTEGRATION POINT
       num_int = NUMBER OF INTEGRATION POINTS, EQ. 1 OR 8

                        Updated 4/7/00
*/

	double ra[]={-0.50, 0.50, 0.50,-0.50,-0.50, 0.50, 0.50,-0.50};
	double sa[]={-0.50,-0.50, 0.50, 0.50,-0.50,-0.50, 0.50, 0.50};
	double ta[]={-0.50,-0.50,-0.50,-0.50, 0.50, 0.50, 0.50, 0.50};
	double temps,tempr,tempt,r,s,t;
	int i,j,k;

	for( k = 0; k < num_int; ++k )
	{
		*(w+k)=1.0;
	}
	for( k = 0; k < num_int; ++k )
	{
		r=g*(*(ra+k));
		s=g*(*(sa+k));
		t=g*(*(ta+k));
		for( i = 0; i < npel ; ++i )
		{
		   tempr=pt5+*(ra+i)*r;
		   temps=pt5+*(sa+i)*s;
		   tempt=pt5+*(ta+i)*t;
		   *(shl+npel*(nsd+1)*k+i)=*(ra+i)*temps*tempt;
		   *(shl+npel*(nsd+1)*k+npel*1+i)=tempr*(*(sa+i))*tempt;
		   *(shl+npel*(nsd+1)*k+npel*2+i)=tempr*temps*(*(ta+i));
		   *(shl+npel*(nsd+1)*k+npel*3+i)=tempr*temps*tempt;
		}
		/*printf("\n");*/
	}
	return 1;
}

int brshl_node2(double *shl_node2)
{
/* 
   This subroutine is a condensed version of the shape function shl_node.  I have
   removed all of the zero terms which made up 75% of all the values in shl_node.

                        Updated 4/7/00
*/

/* For node 0 */
	*(shl_node2) =   -0.50; *(shl_node2+1) =  0.50; *(shl_node2+2) = -0.50;
	*(shl_node2+3) =  0.50; *(shl_node2+4) = -0.50; *(shl_node2+5) =  0.50;

/* For node 1 */
	*(shl_node2+6) = -0.50; *(shl_node2+7)  =  0.50; *(shl_node2+8)  = -0.50;
	*(shl_node2+9) =  0.50; *(shl_node2+10) = -0.50; *(shl_node2+11) =  0.50;

/* For node 2 */
	*(shl_node2+12) =  0.50; *(shl_node2+13) = -0.50; *(shl_node2+14) = -0.50;
	*(shl_node2+15) =  0.50; *(shl_node2+16) = -0.50; *(shl_node2+17) =  0.50;

/* For node 3 */
	*(shl_node2+18) =  0.50; *(shl_node2+19) = -0.50; *(shl_node2+20) = -0.50;
	*(shl_node2+21) =  0.50; *(shl_node2+22) = -0.50; *(shl_node2+23) =  0.50;

/* For node 4 */
	*(shl_node2+24) = -0.50; *(shl_node2+25) =  0.50; *(shl_node2+26) = -0.50;
	*(shl_node2+27) =  0.50; *(shl_node2+28) = -0.50; *(shl_node2+29) =  0.50;

/* For node 5 */
	*(shl_node2+30) = -0.50; *(shl_node2+31) =  0.50; *(shl_node2+32) = -0.50;
	*(shl_node2+33) =  0.50; *(shl_node2+34) = -0.50; *(shl_node2+35) =  0.50;

/* For node 6 */
	*(shl_node2+36) =  0.50; *(shl_node2+37) = -0.50; *(shl_node2+38) = -0.50;
	*(shl_node2+39) =  0.50; *(shl_node2+40) = -0.50; *(shl_node2+41) =  0.50;

/* For node 7 */
	*(shl_node2+42) =  0.50; *(shl_node2+43) = -0.50; *(shl_node2+44) = -0.50;
	*(shl_node2+45) =  0.50; *(shl_node2+46) = -0.50; *(shl_node2+47) =  0.50;

	return 1;
}


int brshg( double *det, int el, double *shl, double *shg, double *xl)
{
/*
     This subroutine calculates the global shape function derivatives for
     a brick element at the gauss points.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det+k)  = JACOBIAN DETERMINANT
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+i) = X-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*1+i) = Y-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*2+i) = Z-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*3+i) = shl(npel*(nsd+1)*k+npel*3+i)
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 8

                        Updated 9/25/01
*/

	double xs[9],temp[9],col1[nsd],col2[nsd];
	int check,i,j,k;

	memcpy(shg,shl,sosh*sizeof(double));

	for( k = 0; k < num_int; ++k )
	{

/* The jacobian, dx/dc, is calculated below */

	   for( j = 0; j < nsd; ++j )
	   {
		for( i = 0; i < nsd; ++i )
		{
		   check=dotX((xs+nsd*i+j),(shg+npel*(nsd+1)*k+npel*j),
			(xl+npel*i),npel);
		}
	   }

	   *(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
	   *(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
	   *(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

	   *(det+k)=*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
	   /*printf("%d %f\n", k, *(det+k));*/

	   if(*(det+k) <= 0.0 ) 
	   {
		printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+k),k);
		return 0;
	   }

/* The inverse of the jacobian, dc/dx, is calculated below */

	   *(temp+1)=*(xs+7)*(*(xs+2))-*(xs+1)*(*(xs+8));
	   *(temp+4)=*(xs)*(*(xs+8))-*(xs+6)*(*(xs+2));
	   *(temp+7)=*(xs+6)*(*(xs+1))-*(xs)*(*(xs+7));
	   *(temp+2)=*(xs+1)*(*(xs+5))-*(xs+4)*(*(xs+2));
	   *(temp+5)=*(xs+3)*(*(xs+2))-*(xs)*(*(xs+5));
	   *(temp+8)=*(xs)*(*(xs+4))-*(xs+3)*(*(xs+1));

	   for( j = 0; j < nsd; ++j )
	   {
		for( i = 0; i < nsd; ++i )
		{
		   *(xs+nsd*i+j)=*(temp+nsd*i+j)/(*(det+k));
		}
	   }
	   for( i = 0; i < npel; ++i )
	   {
		*(col1)=*(shg+npel*(nsd+1)*k+i);
		*(col1+1)=*(shg+npel*(nsd+1)*k+npel*1+i);
		*(col1+2)=*(shg+npel*(nsd+1)*k+npel*2+i);
		*(col2)=*(xs);
		*(col2+1)=*(xs+nsd*1);
		*(col2+2)=*(xs+nsd*2);
		check=dotX((shg+npel*(nsd+1)*k+i),(col2),(col1),nsd);
		*(col2)=*(xs+1);
		*(col2+1)=*(xs+nsd*1+1);
		*(col2+2)=*(xs+nsd*2+1);
		check=dotX((shg+npel*(nsd+1)*k+npel*1+i),(col2),(col1),nsd);
		*(col2)=*(xs+2);
		*(col2+1)=*(xs+nsd*1+2);
		*(col2+2)=*(xs+nsd*2+2);
		check=dotX((shg+npel*(nsd+1)*k+npel*2+i),(col2),(col1),nsd);
		/*printf("%d %f %f %f\n", i, *(shg+npel*(nsd+1)*k+i),
			*(shg+npel*(nsd+1)*k+npel*1+i), 
			*(shg+npel*(nsd+1)*k+npel*2+i));*/
	   }
	}
	return 1; 
}


int brshg_mass( double *det, int el, double *shg, double *xl)
{
/*
     This subroutine calculates the determinant for
     a brick element at the gauss points for the calculation of
     the global mass matrix.  Unlike brshg, we do not have to
     calculate the shape function derivatives with respect to
     the global coordinates x, y, z which are only needed for
     strains and stresses.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det+k)  = JACOBIAN DETERMINANT
       *(shg+npel*(nsd+1)*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*2+i) = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*3+i) = shl(npel*(nsd+1)*k+npel*3+i)
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 8

                        Updated 9/25/01
*/

	double xs[9],temp[9],col1[nsd],col2[nsd];
	int check,i,j,k;

	for( k = 0; k < num_int; ++k )
	{

/* The jacobian, dx/dc, is calculated below */

	   for( j = 0; j < nsd; ++j )
	   {
		for( i = 0; i < nsd; ++i )
		{
		   check=dotX((xs+nsd*i+j),(shg+npel*(nsd+1)*k+npel*j),
			(xl+npel*i),npel);
		}
	   }

	   *(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
	   *(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
	   *(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

	   *(det+k)=*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
	   /*printf(" element %d int. pt. %d determinant %f\n", el, k, *(det+k));*/

	   if(*(det+k) <= 0.0 ) 
	   {
		printf("the element (%d) is inverted; det:%f; integ pt.:%d\n",
			el,*(det+k),k);
		return 0;
	   }
	}
	return 1; 
}


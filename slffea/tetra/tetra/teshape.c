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
#include "teconst.h"

int dotX(double *,double *, double *, int );

int teshl( int gauss_stress_flag, double *shl, double *w)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a tetrahedral element at the gauss points.

     It is based on the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

     To see the shape functions for tetrahedrons, look on page 170.

     Also, look on page 13-4 to 13-6 in the "ANSYS User's Manual".  It
     discusses integration points for tetrahedrons.

     It should be noted that I have permutated the node sequences given
     by Hughes so that in terms of the tetrahedrons given in Figure 3.I.8
     on page 170, node 4 in Hughes is node 1 in SLFFEA, node 1 is node 2,
     and node 2 goes to node 3, and node 3 is node 4.  This is because
     I want to maintain the counter-clockwise node numbering similar to
     the brick.  Hughes reverses this order.

 ....  CALCULATE INTEGRATION-RULE WEIGHTS, SHAPE FUNCTIONS AND
       LOCAL DERIVATIVES FOR A EIGHT-NODE HEXAHEDRAL ELEMENT
       r, s, t = LOCAL ELEMENT COORDINATES ("r","s","t", RESP.)
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("t") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(w+k)    = INTEGRATION-RULE WEIGHT
       i       = LOCAL NODE NUMBER
       k       = INTEGRATION POINT
       num_int = NUMBER OF INTEGRATION POINTS, EQ. 1 OR 8

                        Updated 11/19/01
*/

	double ra[]={ pt1382, pt5854, pt1382, pt1382};
	double sa[]={ pt1382, pt1382, pt5854, pt1382};
	double ta[]={ pt1382, pt1382, pt1382, pt5854};
	double temps,tempr,tempt,r,s,t, fdum;
	int i,j,k;

	if( !gauss_stress_flag )
	{

/* Set ra, sa, ta for shl_node calculation */

		*(ra) = 0.0;     *(sa) = 0.0;     *(ta) = 0.0;
		*(ra + 1) = 1.0; *(sa + 1) = 0.0; *(ta + 1) = 0.0;
		*(ra + 2) = 0.0; *(sa + 2) = 1.0; *(ta + 2) = 0.0;
		*(ra + 3) = 0.0; *(sa + 3) = 0.0; *(ta + 3) = 1.0;
	}

	for( k = 0; k < num_int; ++k )
	{
/* calculating the weights and local dN/ds,dr matrix */

		*(w+k)=0.25;

		r=*(ra+k);
		s=*(sa+k);
		t=*(ta+k);
		fdum = (1.0 - r - s - t);

/* dN/dr */
		*(shl+npel*(nsd+1)*k)= -1.0;                     /* node 0 */
		*(shl+npel*(nsd+1)*k+1)=  1.0;                   /* node 1 */
		*(shl+npel*(nsd+1)*k+2)=  0.0;                   /* node 2 */
		*(shl+npel*(nsd+1)*k+3)=  0.0;                   /* node 3 */

/* dN/ds */
		*(shl+npel*(nsd+1)*k+npel*1)= -1.0;              /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*1+1)=  0.0;            /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*1+2)=  1.0;            /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*1+3)=  0.0;            /* node 3 */

/* dN/dt */
		*(shl+npel*(nsd+1)*k+npel*2)= -1.0;              /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*2+1)=  0.0;            /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*2+2)=  0.0;            /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*2+3)=  1.0;            /* node 3 */

/* N */
		*(shl+npel*(nsd+1)*k+npel*3)= fdum;              /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*3+1)= r;               /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*3+2)= s;               /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*3+3)= t;               /* node 3 */

		/*printf("\n");*/

		/*for( i = 0; i < nsd+1; ++i )
		{
		    for( j = 0; j < npel; ++j )
		    {
			printf(" %14.6e",*(shl+npel*(nsd+1)*k+npel*i + j));
		    }
		    printf("\n");
		}
		printf("\n");*/
	}
	return 1;
}

int teshg( double *det, int el, double *shl, double *shg, double *xl)
{
/*
     This subroutine calculates the global shape function derivatives for
     a tetrahedral at the gauss points.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det)  = JACOBIAN DETERMINANT
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("t") DERIVATIVE OF SHAPE FUNCTION
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

                        Updated 11/19/01
*/

	double xs[9],temp[9],col1[nsd],col2[nsd];
	int check,i,j,k;

	memcpy(shg,shl,sosh*sizeof(double));

/* The jacobian, dx/dc, is calculated below */

#if 0
	for( j = 0; j < nsd; ++j )
	{
		for( i = 0; i < nsd; ++i )
		{
		   check=dotX((xs+nsd*i+j),(shg+npel*j), (xl+npel*i),npel);
		}
	}
#endif

#if 1
	*(xs)   = *(xl + 1) - *(xl);
	*(xs+1) = *(xl + 2) - *(xl);
	*(xs+2) = *(xl + 3) - *(xl);
	*(xs+3) = *(xl + 5) - *(xl + 4);
	*(xs+4) = *(xl + 6) - *(xl + 4);
	*(xs+5) = *(xl + 7) - *(xl + 4);
	*(xs+6) = *(xl + 9) - *(xl + 8);
	*(xs+7) = *(xl + 10) - *(xl + 8);
	*(xs+8) = *(xl + 11) - *(xl + 8);
#endif

	*(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
	*(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
	*(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

	*(det)=*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));

	/* *(det)=fabs(*(det));*/

	/*printf("%d %f\n", k, *(det));*/

	if(*(det) < 0.0 ) 
	{
		printf("the element (%d) is inverted; det:%f\n", el,*(det));
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
		   *(xs+nsd*i+j)=*(temp+nsd*i+j)/(*(det));
		}
	}

	for( k = 0; k < num_int; ++k )
	{
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



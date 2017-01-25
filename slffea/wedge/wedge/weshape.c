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
#include "weconst.h"

int dotX(double *,double *, double *, int );

int weshl( double g, double *shl, double *w)
{
/* 
     This subroutine calculates the local shape function derivatives for
     a wedge element at the gauss points.

     It is based on the subroutine QDCSHL from the book
     "The Finite Element Method" by Thomas Hughes, page 784.

     To see the shape functions for wedges, look on page 171-172.

     Also, look on page 13-3 to 13-6 in the "ANSYS User's Manual".  It
     discusses integration points for triangles.

     It should be noted that I have permutated the node sequences
     given by Hughes so that in terms of the triangle given in
     Figure 3.I.5 on page 167, node 3 in Hughes is node 1 in SLFFEA,
     node 2 is node 3, and node 1 goes to node 2.  This is because
     I want to be consistant with the tetrahedron.  You can read
     more about this change in teshl for the tetrahedron.

 ....  CALCULATE INTEGRATION-RULE WEIGHTS, SHAPE FUNCTIONS AND
       LOCAL DERIVATIVES FOR A EIGHT-NODE HEXAHEDRAL ELEMENT
       r, s = LOCAL ELEMENT COORDINATES along triangle faces 
       xi = LOCAL ELEMENT COORDINATES along length of body
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(w+k)    = INTEGRATION-RULE WEIGHT
       i       = LOCAL NODE NUMBER
       k       = INTEGRATION POINT
       num_int = NUMBER OF INTEGRATION POINTS, EQ. 1 OR 6

                        Updated 11/18/01
*/

	double ra[]={ pt1667, pt6667, pt1667, pt1667, pt6667, pt1667};
	double sa[]={ pt1667, pt1667, pt6667, pt1667, pt1667, pt6667};
	double ta[]={-0.50,-0.50,-0.50, 0.50, 0.50, 0.50};
	double temps,tempr,tempt,r,s,t,fdum;
	int i,j,k;


	for( k = 0; k < num_int; ++k )
	{
		*(w+k)=pt3333;
	}

	if( g > 1.99 )
	{

/* Set ra, sa, ta for shl_node calculation */

		*(ra) = 0.0;     *(sa) = 0.0;
		*(ra + 1) = 1.0; *(sa + 1) = 0.0;
		*(ra + 2) = 0.0; *(sa + 2) = 1.0;
		*(ra + 3) = 0.0; *(sa + 3) = 0.0;
		*(ra + 4) = 1.0; *(sa + 4) = 0.0;
		*(ra + 5) = 0.0; *(sa + 5) = 1.0;
	}

	for( k = 0; k < num_int; ++k )
	{
		r=*(ra+k);
		s=*(sa+k);
		t=g*(*(ta+k));
		fdum = (1.0 - r - s);

/* dN/dr */
		*(shl+npel*(nsd+1)*k)  = -0.5*( 1.0 - t);            /* node 0 */
		*(shl+npel*(nsd+1)*k+1)=  0.5*( 1.0 - t);            /* node 1 */
		*(shl+npel*(nsd+1)*k+2)=  0.0;                       /* node 2 */
		*(shl+npel*(nsd+1)*k+3)= -0.5*( 1.0 + t);            /* node 3 */
		*(shl+npel*(nsd+1)*k+4)=  0.5*( 1.0 + t);            /* node 4 */
		*(shl+npel*(nsd+1)*k+5)=  0.0;                       /* node 5 */

/* dN/ds */
		*(shl+npel*(nsd+1)*k+npel*1)  = -0.5*( 1.0 - t);     /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*1+1)=  0.0;                /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*1+2)=  0.5*( 1.0 - t);     /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*1+3)= -0.5*( 1.0 + t);     /* node 3 */
		*(shl+npel*(nsd+1)*k+npel*1+4)=  0.0;                /* node 4 */
		*(shl+npel*(nsd+1)*k+npel*1+5)=  0.5*( 1.0 + t);     /* node 5 */

/* dN/dxi (dN/dt) */
		*(shl+npel*(nsd+1)*k+npel*2)  = -0.5*fdum;           /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*2+1)= -0.5*r;              /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*2+2)= -0.5*s;              /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*2+3)=  0.5*fdum;           /* node 3 */
		*(shl+npel*(nsd+1)*k+npel*2+4)=  0.5*r;              /* node 4 */
		*(shl+npel*(nsd+1)*k+npel*2+5)=  0.5*s;              /* node 5 */

/* N */
		*(shl+npel*(nsd+1)*k+npel*3)  = 0.5*fdum*( 1.0 - t);  /* node 0 */
		*(shl+npel*(nsd+1)*k+npel*3+1)= 0.5*r*( 1.0 - t);     /* node 1 */
		*(shl+npel*(nsd+1)*k+npel*3+2)= 0.5*s*( 1.0 - t);     /* node 2 */
		*(shl+npel*(nsd+1)*k+npel*3+3)= 0.5*fdum*( 1.0 + t);  /* node 3 */
		*(shl+npel*(nsd+1)*k+npel*3+4)= 0.5*r*( 1.0 + t);     /* node 4 */
		*(shl+npel*(nsd+1)*k+npel*3+5)= 0.5*s*( 1.0 + t);     /* node 5 */

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

int weshl_node2(double *shl_node2)
{
/* 
   This subroutine is a condensed version of the shape function shl_node.  I have
   removed all of the zero terms which made up 56% of all the values in shl_node.

			Updated 11/19/01
*/

/* For node 0 */
	*(shl_node2) =   -1.00; *(shl_node2+1) =  1.00; *(shl_node2+2) = -1.00;
	*(shl_node2+3) =  1.00; *(shl_node2+4) = -0.50; *(shl_node2+5) =  0.50;

/* For node 1 */
	*(shl_node2+6) = -1.00; *(shl_node2+7)  =  1.00; *(shl_node2+8)  = -1.00;
	*(shl_node2+9) =  1.00; *(shl_node2+10) = -0.50; *(shl_node2+11) =  0.50;

/* For node 2 */
	*(shl_node2+12) = -1.00; *(shl_node2+13) =  1.00; *(shl_node2+14) = -1.00;
	*(shl_node2+15) =  1.00; *(shl_node2+16) = -0.50; *(shl_node2+17) =  0.50;

/* For node 3 */
	*(shl_node2+18) = -1.00; *(shl_node2+19) =  1.00; *(shl_node2+20) = -1.00;
	*(shl_node2+21) =  1.00; *(shl_node2+22) = -0.50; *(shl_node2+23) =  0.50;

/* For node 4 */
	*(shl_node2+24) = -1.00; *(shl_node2+25) =  1.00; *(shl_node2+26) = -1.00;
	*(shl_node2+27) =  1.00; *(shl_node2+28) = -0.50; *(shl_node2+29) =  0.50;

/* For node 5 */
	*(shl_node2+30) = -1.00; *(shl_node2+31) =  1.00; *(shl_node2+32) = -1.00;
	*(shl_node2+33) =  1.00; *(shl_node2+34) = -0.50; *(shl_node2+35) =  0.50;

	return 1;
}


int weshg( double *det, int el, double *shl, double *shg, double *xl)
{
/*
     This subroutine calculates the global shape function derivatives for
     a wedge at the gauss points.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det+k)  = JACOBIAN DETERMINANT
       *(shl+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*2+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl+npel*(nsd+1)*k+npel*3+i) = LOCAL SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+i) = X-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*1+i) = Y-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*2+i) = Z-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*3+i) = shl(npel*(nsd+1)*k+npel*3+i)
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 6

                        Updated 4/7/01
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


int weshg_mass( double *det, int el, double *shg, double *xl)
{
/*
     This subroutine calculates the determinant for
     a wedge element at the gauss points for the calculation of
     the global mass matrix.  Unlike weshg, we do not have to
     calculate the shape function derivatives with respect to
     the global coordinates x, y, z which are only needed for
     strains and stresses.

     It is based on the subroutine QDCSHG from the book
     "The Finite Element Method" by Thomas Hughes, page 783.

 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

       *(xl+j+npel*i) = GLOBAL COORDINATES
       *(det+k)  = JACOBIAN DETERMINANT
       *(shg+npel*(nsd+1)*k+i) = LOCAL ("r") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*1+i) = LOCAL ("s") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*2+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*3+i) = shl(npel*(nsd+1)*k+npel*3+i)
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 6

                        Updated 4/7/00
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


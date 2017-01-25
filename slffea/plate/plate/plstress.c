/*
    This subroutine calculates the global shape function derivatives for
    a quadrilateral element at the nodal points.

        Updated 9/4/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plconst.h"

extern int sof;

int dotX(double *,double *, double *, int );

int plstress_shg(double *det, int el, double *shl_node2, double *shg, double *xl )
{
/*
 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

   *(xl+npel*(0,1,2)+*(node_xi+2*k+(0,1))) = GLOBAL COORDINATES CORRESPONDING TO
                                             NONZERO SHAPE FUNCTION dN/dxi
   *(xl+npel*(0,1,2)+*(node_eta+2*k+(0,1))) = GLOBAL COORDINATES CORRESPONDING TO
                                              NONZERO SHAPE FUNCTION dN/deta
       *(det+k)  = JACOBIAN DETERMINANT
       *(shl_node2+2*nsd2*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl_node2+2*nsd2*k+2*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl_node2+2*nsd2*k+2*2+i) = LOCAL SHAPE FUNCTION

       2 has replaced "npel" because there are only 2 non-zero shape function
       derivatives when evaluating at a node.
       
       *(shg+npel*(nsd2+1)*k+i) = X-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*1+i) = Y-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd2+1)*k+npel*2+i) = shl(npel*(nsd2+1)*k+npel*3+i)
       *(xs+nsd2*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 8

       shl_node2 = derivative of shape function dN/dc evaluated at nodal point
       node_xi[] = only nonzero shape function number at particular node for dN/dxi
       node_eta[] = only nonzero shape function number at particular node
                    for dN/deta

                1111
                        Updated 9/4/06
*/

	int node_xi[]  = {0,1,0,1,2,3,2,3};
	int node_eta[] = {0,3,1,2,1,2,0,3};

	double xs[4],temp;
	int check,i,j,k;

	memset(shg,0,soshb*sof);

	for( k = 0; k < npel; ++k )
	{

/* The jacobian, dx/dc, is calculated below */

	   for( j = 0; j < nsd2; ++j )
	   {
		*(xs+nsd2*j) = 
		   *(shl_node2+2*nsd2*k)*(*(xl+npel*j+*(node_xi+2*k))) +
		   *(shl_node2+2*nsd2*k+1)*(*(xl+npel*j+*(node_xi+2*k+1)));
		*(xs+nsd2*j+1) = 
		   *(shl_node2+2*nsd2*k+2*1)*(*(xl+npel*j+*(node_eta+2*k))) +
		   *(shl_node2+2*nsd2*k+2*1+1)*(*(xl+npel*j+*(node_eta+2*k+1)));
	   }

	   *(det+k)=*(xs)*(*(xs+3))-*(xs+2)*(*(xs+1));
	   /*printf("%d %f\n", k, *(det+k));*/

	   if(*(det+k) < 0.0 ) 
	   {
		printf("the element (%6d) is inverted: %f %d\n", el,*(det+k),k);
		return 1;
	   }

/* The inverse of the jacobian, dc/dx, is calculated below */

	   temp=*(xs);
	   *(xs)=*(xs+3)/(*(det+k));
	   *(xs+1)*=-1./(*(det+k));
	   *(xs+2)*=-1./(*(det+k));
	   *(xs+3)=temp/(*(det+k));

	   *(shg+npel*(nsd2+1)*k+*(node_xi+2*k)) =
		*(shl_node2+4*k)*(*(xs));
	   *(shg+npel*(nsd2+1)*k+npel*1+*(node_xi+2*k)) =
		*(shl_node2+4*k)*(*(xs+1));

	   *(shg+npel*(nsd2+1)*k+*(node_xi+2*k+1)) =
		*(shl_node2+4*k+1)*(*(xs));
	   *(shg+npel*(nsd2+1)*k+npel*1+*(node_xi+2*k+1)) =
		*(shl_node2+4*k+1)*(*(xs+1));

	   *(shg+npel*(nsd2+1)*k+*(node_eta+2*k)) +=
		*(shl_node2+4*k+2*1)*(*(xs+2));
	   *(shg+npel*(nsd2+1)*k+npel*1+*(node_eta+2*k)) +=
		*(shl_node2+4*k+2*1)*(*(xs+3));

	   *(shg+npel*(nsd2+1)*k+*(node_eta+2*k+1)) +=
		*(shl_node2+4*k+2*1+1)*(*(xs+2));
	   *(shg+npel*(nsd2+1)*k+npel*1+*(node_eta+2*k+1)) +=
		*(shl_node2+4*k+2*1+1)*(*(xs+3));

	}

	*(shg+npel*2) = 1.0;
	*(shg+npel*(nsd2+1)*1+npel*2 + 1) = 1.0;
	*(shg+npel*(nsd2+1)*2+npel*2 + 2) = 1.0;
	*(shg+npel*(nsd2+1)*3+npel*2 + 3) = 1.0;

	return 1; 
}


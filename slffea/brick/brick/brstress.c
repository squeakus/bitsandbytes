/*
    This subroutine calculates the global shape function derivatives for
    a brick element at the nodal points.  It is streamlined by removal
    of the zero terms.

        Updated 9/4/01

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
#include "brconst.h"

extern int sof;

int brstress_shg(double *det, int el, double *shl_node2, double *shg, double *xl )
{
/*
 ....  CALCULATE GLOBAL DERIVATIVES OF SHAPE FUNCTIONS AND
       JACOBIAN DETERMINANTS FOR AN EIGHT-NODE HEXAHEDRAL ELEMENT

   *(xl+npel*(0,1,2)+*(node_xi+2*k+(0,1))) = GLOBAL COORDINATES CORRESPONDING TO
                                             NONZERO SHAPE FUNCTION dN/dxi
   *(xl+npel*(0,1,2)+*(node_eta+2*k+(0,1))) = GLOBAL COORDINATES CORRESPONDING TO
                                              NONZERO SHAPE FUNCTION dN/deta
   *(xl+npel*(0,1,2)+*(node_zeta+2*k+(0,1))) = GLOBAL COORDINATES CORRESPONDING TO
                                               NONZERO SHAPE FUNCTION dN/dzeta
       *(det+k)  = JACOBIAN DETERMINANT
       *(shl_node2+2*nsd*k+i) = LOCAL ("XI") DERIVATIVE OF SHAPE FUNCTION
       *(shl_node2+2*nsd*k+2*1+i) = LOCAL ("ETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl_node2+2*nsd*k+2*2+i) = LOCAL ("ZETA") DERIVATIVE OF SHAPE FUNCTION
       *(shl_node2+2*nsd*k+2*3+i) = LOCAL SHAPE FUNCTION

       2 has replaced "npel" because there are only 2 non-zero shape function
       derivatives when evaluating at a node.
       
       *(shg+npel*(nsd+1)*k+i) = X-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*1+i) = Y-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*2+i) = Z-DERIVATIVE OF SHAPE FUNCTION
       *(shg+npel*(nsd+1)*k+npel*3+i) = shl(npel*(nsd+1)*k+npel*3+i)
       *(xs+nsd*j+i) = JACOBIAN MATRIX
          i    = LOCAL NODE NUMBER OR GLOBAL COORDINATE NUMBER
          j    = GLOBAL COORDINATE NUMBER
          k    = INTEGRATION-POINT NUMBER
       num_int    = NUMBER OF INTEGRATION POINTS, EQ.1 OR 8

       shl_node2 = derivative of shape function dN/dc evaluated at nodal point
       node_xi[] = only nonzero shape function number at particular node
                   for dN/dxi
       node_eta[] = only nonzero shape function number at particular node
                    for dN/deta
       node_zeta[] = only nonzero shape function number at particular node
                     for dN/dzeta

                1111
                        Updated 9/4/06
*/

	int node_xi[]  = {0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7};
	int node_eta[] = {0,3,1,2,1,2,0,3,4,7,5,6,5,6,4,7};
	int node_zeta[]= {0,4,1,5,2,6,3,7,0,4,1,5,2,6,3,7};

	double xs[9],temp[9],col1[nsd],col2[nsd],temp1,temp2;
	int check,i,j,k;

	memset(shg,0,sosh*sof);

	for( k = 0; k < npel; ++k )
	{

/* The jacobian, dx/dc, is calculated below */

	   for( j = 0; j < nsd; ++j )
	   {
		*(xs+nsd*j) = 
		   *(shl_node2+2*nsd*k)*(*(xl+npel*j+*(node_xi+2*k))) +
		   *(shl_node2+2*nsd*k+1)*(*(xl+npel*j+*(node_xi+2*k+1)));
		*(xs+nsd*j+1) = 
		   *(shl_node2+2*nsd*k+2*1)*(*(xl+npel*j+*(node_eta+2*k))) +
		   *(shl_node2+2*nsd*k+2*1+1)*(*(xl+npel*j+*(node_eta+2*k+1)));
		*(xs+nsd*j+2) = 
		   *(shl_node2+2*nsd*k+2*2)*(*(xl+npel*j+*(node_zeta+2*k))) +
		   *(shl_node2+2*nsd*k+2*2+1)*(*(xl+npel*j+*(node_zeta+2*k+1)));
	   }

	   *(temp)=*(xs+4)*(*(xs+8))-*(xs+7)*(*(xs+5));
	   *(temp+3)=*(xs+6)*(*(xs+5))-*(xs+3)*(*(xs+8));
	   *(temp+6)=*(xs+3)*(*(xs+7))-*(xs+6)*(*(xs+4));

	   *(det+k)=*(xs)*(*(temp))+*(xs+1)*(*(temp+3))+*(xs+2)*(*(temp+6));
	   /*printf("%d %f\n", k, *(det+k));*/

	   if(*(det+k) <= 0.0 ) 
	   {
		printf("the element (%6d) is inverted: %f %d\n", el,*(det+k),k);
		return 1;
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

	   *(shg+npel*(nsd+1)*k+*(node_xi+2*k)) =
		*(shl_node2+6*k)*(*(xs));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_xi+2*k)) =
		*(shl_node2+6*k)*(*(xs+1));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_xi+2*k)) =
		*(shl_node2+6*k)*(*(xs+2));

	   *(shg+npel*(nsd+1)*k+*(node_xi+2*k+1)) =
		*(shl_node2+6*k+1)*(*(xs));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_xi+2*k+1)) =
		*(shl_node2+6*k+1)*(*(xs+1));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_xi+2*k+1)) =
		*(shl_node2+6*k+1)*(*(xs+2));

	   *(shg+npel*(nsd+1)*k+*(node_eta+2*k)) +=
		*(shl_node2+6*k+2*1)*(*(xs+3));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_eta+2*k)) +=
		*(shl_node2+6*k+2*1)*(*(xs+4));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_eta+2*k)) +=
		*(shl_node2+6*k+2*1)*(*(xs+5));

	   *(shg+npel*(nsd+1)*k+*(node_eta+2*k+1)) +=
		*(shl_node2+6*k+2*1+1)*(*(xs+3));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_eta+2*k+1)) +=
		*(shl_node2+6*k+2*1+1)*(*(xs+4));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_eta+2*k+1)) +=
		*(shl_node2+6*k+2*1+1)*(*(xs+5));

	   *(shg+npel*(nsd+1)*k+*(node_zeta+2*k)) +=
		*(shl_node2+6*k+2*2)*(*(xs+6));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_zeta+2*k)) +=
		*(shl_node2+6*k+2*2)*(*(xs+7));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_zeta+2*k)) +=
		*(shl_node2+6*k+2*2)*(*(xs+8));

	   *(shg+npel*(nsd+1)*k+*(node_zeta+2*k+1)) +=
		*(shl_node2+6*k+2*2+1)*(*(xs+6));
	   *(shg+npel*(nsd+1)*k+npel*1+*(node_zeta+2*k+1)) +=
		*(shl_node2+6*k+2*2+1)*(*(xs+7));
	   *(shg+npel*(nsd+1)*k+npel*2+*(node_zeta+2*k+1)) +=
		*(shl_node2+6*k+2*2+1)*(*(xs+8));
	}
	return 1; 
}


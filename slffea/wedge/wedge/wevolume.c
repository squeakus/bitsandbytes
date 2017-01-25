/*
    This utility function calculates the Volume of each
    wedge element using shape functions and gaussian
    integration.

                Updated 10/4/01

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

extern int numel;
extern double shg[sosh], shl[sosh], w[num_int];

int weshg( double *, int, double *, double *, double *);

int weVolume( int *connect, double *coord, double *Vol)
{
	double coord_el_trans[npel*nsd];
	int i, j, k, node, check;
	double det[num_int];

	for( k = 0; k < numel; ++k )
	{

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node=*(connect+npel*k+j);

			*(coord_el_trans+j)=*(coord+nsd*node);
			*(coord_el_trans+npel*1+j)=*(coord+nsd*node+1);
			*(coord_el_trans+npel*2+j)=*(coord+nsd*node+2);
		}

		check = weshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with weshg \n");

/* Calculate the Volume from determinant of the Jacobian */

		for( j = 0; j < num_int; ++j )
		{
/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in 
   "The Finite Element Method" by Thomas Hughes, page 174
*/
			*(Vol + k) += 0.5*(*(w+j))*(*(det+j));
		}
	}

	return 1;
}

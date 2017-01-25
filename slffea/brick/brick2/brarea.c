/*
    This utility function calculates the Area of each convection
    film surfaces for the thermal brick element using shape functions
    and gaussian integration.

                Updated 9/27/01

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

extern int numel_film;
extern double shl_film[sosh_film], w[num_int];

int brshface( double *, int , double *, double *);

int brArea( int *connect_film, double *coord, double *Area )
{
	double coord_el_trans[npel*nsd];
	int i, j, k, node, check;
	double dArea[num_int_film];

	for( k = 0; k < numel_film; ++k )
	{

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel_film; ++j )
		{
			node = *(connect_film+npel_film*k+j);

			*(coord_el_trans+j)=*(coord+nsd*node);
			*(coord_el_trans+npel_film*1+j)=*(coord+nsd*node+1);
			*(coord_el_trans+npel_film*2+j)=*(coord+nsd*node+2);
		}

		check = brshface( dArea, k, shl_film, coord_el_trans);
		if(!check) printf( "Problems with brshface \n");

/* Calculate the Area from determinant of the Jacobian */

		for( j = 0; j < num_int_film; ++j )
		{
			*(Area + k) += *(w+j)*(*(dArea+j));
		}
	}

	return 1;
}

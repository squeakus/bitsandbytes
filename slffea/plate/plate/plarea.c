/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a plate.

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
#include "plconst.h"
#include "plstruct.h"

extern int numel;
extern SH shg, shl;
extern double w[num_int+1];

int plshg( double *, int, SH, SH, double *);

int plArea( int *connect, double *coord, double *Area)
{
	double coord_el_trans[npel*nsd];
        int i, j, k, node, check;
	double det[num_int+1];

        for( k = 0; k < numel; ++k )
        {

/* Create the coord_el transpose vector for one element */

                for( j = 0; j < npel; ++j )
                {
			node = *(connect+npel*k+j);

                        *(coord_el_trans+j)=*(coord+nsd*node);
                        *(coord_el_trans+npel*1+j)=*(coord+nsd*node+1);
		}

		check=plshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with plshg \n");

                for( j = 0; j < num_int; ++j )
                {
			*(Area + k) += *(w+j)*(*(det+j));
                }
        }

	return 1;
}


/*
    This utility function calculates the Length of each
    truss element.

                Updated 1/24/03

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
#include "../truss/tsconst.h"

extern int numel;

int tsLength( int *connect, double *coord, double *length)
{
	int i, j, k, node0, node1, check;
	double Lx, Ly, Lz, Lsq;

	for( k = 0; k < numel; ++k )
	{

                node0 = *(connect+k*npel);
                node1 = *(connect+k*npel+1);

/* Create the length vector for one element */

                Lx = *(coord+nsd*node1) - *(coord+nsd*node0);
                Ly = *(coord+nsd*node1+1) - *(coord+nsd*node0+1);
                Lz = *(coord+nsd*node1+2) - *(coord+nsd*node0+2);

                Lsq = Lx*Lx + Ly*Ly + Lz*Lz;
                *(length + k) = sqrt(Lsq);
	}

	return 1;
}

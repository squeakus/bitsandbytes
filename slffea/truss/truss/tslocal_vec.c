/*
    This utility function calculates the local x, y, z (called local_xyz[])
    basis vectors for each element for a finite element program which does
    analysis on a 2 node truss element.

		Updated 11/8/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tsconst.h"
#include "tsstruct.h"

extern int numel;

int tslocal_vectors( int *connect, double *coord, double *length , double *local_xyz )
{
/*
    The mechanism below for calculating rotations is based on the method
    givin in the book, "A First Course in the Finite Element
    Method 2nd Ed." by Daryl L. Logan and my own.  See pages 236-239 in:

     Logan, Daryl L., A First Course in the Finite Element Method 2nd Ed., PWS-KENT,
        1992.

*/

	int i, i1, i2, i3, i4, i5, j, k, sof;
	int check, counter, node0, node1;
	double L, Lx, Ly, Lz, Lsq, Lxy, Lxysq;
	double fdum1, fdum2, fdum3, fdum4;
	double coord_el_trans[npel*nsd];

	sof = sizeof(double);

	for( k = 0; k < numel; ++k )
	{
		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);
		Lx = *(coord+nsd*node1) - *(coord+nsd*node0);
		Ly = *(coord+nsd*node1+1) - *(coord+nsd*node0+1);
		Lz = *(coord+nsd*node1+2) - *(coord+nsd*node0+2);

		/*printf(" Lx, Ly, Lz %f %f %f\n ", Lx, Ly, Lz);*/

		Lsq = Lx*Lx+Ly*Ly+Lz*Lz;
		L = sqrt(Lsq);
		Lx /= L; Ly /= L; Lz /= L;
		*(length + k) = L;

		Lxysq = Lx*Lx + Ly*Ly;
		Lxy = sqrt(Lxysq);

		*(local_xyz + nsdsq*k)     = Lx;
		*(local_xyz + nsdsq*k + 1) = Ly;
		*(local_xyz + nsdsq*k + 2) = Lz;
		*(local_xyz + nsdsq*k + 3) = -Ly/Lxy;
		*(local_xyz + nsdsq*k + 4) = Lx/Lxy;
		*(local_xyz + nsdsq*k + 5) = 0.0;
		*(local_xyz + nsdsq*k + 6) = -Lx*Lz/Lxy;
		*(local_xyz + nsdsq*k + 7) = -Ly*Lz/Lxy;
		*(local_xyz + nsdsq*k + 8) = Lxy;

/* If local x axis lies on global z axis, modify rotate. */

		if(Lz > ONE)
		{
			/*printf("%d %f\n", k, *(local_xyz + nsdsq*k + 2)); */
			memset(local_xyz + nsdsq*k,0,nsdsq*sof);
			*(local_xyz + nsdsq*k + 2) = 1.0;
			*(local_xyz + nsdsq*k + 4) = 1.0;
			*(local_xyz + nsdsq*k + 6) = -1.0;
		}

		if(Lz < -ONE)
		{
			/*printf("%d %f\n", k, *(local_xyz + nsdsq*k + 2)); */
			memset(local_xyz + nsdsq*k,0,nsdsq*sof);
			*(local_xyz + nsdsq*k + 2) = -1.0;
			*(local_xyz + nsdsq*k + 4) = 1.0;
			*(local_xyz + nsdsq*k + 6) = 1.0;
		}
	}

	return 1;
}


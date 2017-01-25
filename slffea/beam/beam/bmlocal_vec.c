/*
    This utility function calculates the local x, y, z (called local_xyz[])
    basis vectors for each element for a finite element program which does
    analysis on a 2 node beam element.

		Updated 11/7/06

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
#include "bmconst.h"
#include "bmstruct.h"

extern int numel;

/* I use bmnormcrossX rather than normcrossX because the case that the 2
   input vectors are zero is acceptable.  See below to find out what happens
   in this case.
*/

int bmnormcrossX(double *, double *, double *);

int bmlocal_vectors( double *axis_z, int *connect, double *coord, double *length,
	double *local_xyz ) 
{
/*
    The mechanism below for calculating rotations is based on a combination
    of the method givin in the book, "A First Course in the Finite Element
    Method 2nd Ed." by Daryl L. Logan and my own.  See pages 236-239 in:

     Logan, Daryl L., A First Course in the Finite Element Method 2nd Ed., PWS-KENT,
        1992. 

*/

	int i, i1, i2, i3, i4, i5, j, k, sof;
	int check, counter, node0, node1;
	double L, Lx, Ly, Lz, Lsq, axis_x[nsd], axis_y[nsd];
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
		*(axis_x) = Lx;
		*(axis_x+1) = Ly;
		*(axis_x+2) = Lz;
		*(length + k) = L;

/* To find axis_y, take cross product of axis_z and axis_x */

		check = bmnormcrossX((axis_z+nsd*k), axis_x, axis_y);
		if(!check)
		{

/* If magnitude of axis_y < SMALL(i.e. axis_z and axis_x are parallel), then make
local x global z, local z global x, and local y global y.  */

		   memset(local_xyz + nsdsq*k,0,nsdsq*sof);
		   *(local_xyz + nsdsq*k + 2) = 1.0;
		   *(local_xyz + nsdsq*k + 4) = 1.0;
		   *(local_xyz + nsdsq*k + 6) = -1.0;
		   if(Lz < -ONE)
		   {
			memset(local_xyz + nsdsq*k,0,nsdsq*sof);
			*(local_xyz + nsdsq*k + 2) = -1.0;
			*(local_xyz + nsdsq*k + 4) = 1.0;
			*(local_xyz + nsdsq*k + 6) = 1.0;
		   }
		   *(axis_z + nsd*k) = *(local_xyz + nsdsq*k + 6);
		   *(axis_z + nsd*k + 1) = 0.0;
		   *(axis_z + nsd*k + 2) = 0.0;
		}
		else
		{

/* To find the true axis_z, take cross product of axis_x and axis_y */

		   check = bmnormcrossX(axis_x, axis_y, (axis_z+nsd*k));
		   if(!check) printf( "Problems with bmnormcrossX \n");

/* Assembly of the 3X3 rotation matrix for the 12X12 global rotation
   matrix */
		   *(local_xyz + nsdsq*k) = *(axis_x);
		   *(local_xyz + nsdsq*k+1) = *(axis_x+1);
		   *(local_xyz + nsdsq*k+2) = *(axis_x+2);
		   *(local_xyz + nsdsq*k+3) = *(axis_y);
		   *(local_xyz + nsdsq*k+4) = *(axis_y+1);
		   *(local_xyz + nsdsq*k+5) = *(axis_y+2);
		   *(local_xyz + nsdsq*k+6) = *(axis_z+nsd*k);
		   *(local_xyz + nsdsq*k+7) = *(axis_z+nsd*k+1);
		   *(local_xyz + nsdsq*k+8) = *(axis_z+nsd*k+2);
		}
	}

	return 1;
}


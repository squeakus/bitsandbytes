/*
    This utility function calculates the local x, y, z (called local_xyz[])
    basis vectors for each element for a finite element program
    which does analysis on a 3 node plate or triangle element.

    For 3-D triangle meshes, I have to rotate from the global coordinates to
    the local x and y coordinates which lie in the plane of the element.
    To do this I have to calculate the normal to the plate face, then
    cross product that normal with an in plane vector to get the other
    local axis direction.

		Updated 11/6/06

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

#define nsd             3                          /* spatial dimensions per node */
#define nsdsq           9                          /* nsd squared */
#define npel3           3                          /* nodes per triangle element */
#define npel4           4                          /* nodes per quad element */

extern int numel;

int normcrossX(double *, double *, double *);

int local_vectors_triangle( int *connect, double *coord, double *local_xyz ) 
{
/*
   Calculation of the basis is taken from:

     Przemieniecki, J. S., Theory of Matrix Structural Analysis, Dover Publications Inc.,
        New York, 1985.

   page 87-89.  The notation is mosty the same except for the fact that I use the name Xpr,
   Ypr, and Zpr rather than Xrp, Yrp, and Zrp to be consistent with the calculation of Xpq,
   Ypq, and Zpq.
*/

	int i, i1, i2, i3, i4, i5, j, k, sof;
	int check, counter, node;
	double fdum1, fdum2, fdum3, fdum4;
	double coord_el_trans[npel3*nsd];
	double local_x[nsd], local_y[nsd], local_z[nsd];
	double X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3, X12, X13, Y12, Y13, Z12, Z13;
	double Xp, Xr, Xq, Yp, Yr, Yq, Zp, Zr, Zq, Xpr, Xpq, Ypr, Ypq, Zpr, Zpq;
	double Lpq, Ltr, Mpq, Mtr, Npq, Ntr, Dpq, Dpt, Dtr;

	sof = sizeof(double);

	for( k = 0; k < numel; ++k )
	{
		for( j = 0; j < npel3; ++j )
		{
			node = *(connect+npel3*k+j);

			*(coord_el_trans + j) = *(coord + nsd*node);
			*(coord_el_trans + npel3*1 + j) = *(coord + nsd*node + 1);
			*(coord_el_trans + npel3*2 + j) = *(coord + nsd*node + 2);
		}

		Xp = *(coord_el_trans);
		Xr = *(coord_el_trans + 1);
		Xq = *(coord_el_trans + 2);
		Xpr = Xr - Xp; Xpq = Xq - Xp;

		Yp = *(coord_el_trans + npel3*1);
		Yr = *(coord_el_trans + npel3*1 + 1);
		Yq = *(coord_el_trans + npel3*1 + 2);
		Ypr = Yr - Yp; Ypq = Yq - Yp;

		Zp = *(coord_el_trans + npel3*2);
		Zr = *(coord_el_trans + npel3*2 + 1);
		Zq = *(coord_el_trans + npel3*2 + 2);
		Zpr = Zr - Zp; Zpq = Zq - Zp;

		Dpq = Xpq*Xpq + Ypq*Ypq + Zpq*Zpq;
		Dpq = sqrt(Dpq);
		Lpq = Xpq/Dpq; Mpq = Ypq/Dpq; Npq = Zpq/Dpq;

		Dpt = Lpq*Xpr + Mpq*Ypr + Npq*Zpr;
		Dtr = Xpr*Xpr + Ypr*Ypr + Zpr*Zpr - Dpt*Dpt;
		Dtr = sqrt(Dtr);

		Ltr = (Xpr - Lpq*Dpt)/Dtr;
		Mtr = (Ypr - Mpq*Dpt)/Dtr;
		Ntr = (Zpr - Npq*Dpt)/Dtr;

		*(local_x)   = Ltr;    *(local_y)   = Lpq;
		*(local_x+1) = Mtr;    *(local_y+1) = Mpq;
		*(local_x+2) = Ntr;    *(local_y+2) = Npq;

/* Calculate the local z basis vector for element k */
		check = normcrossX( local_x, local_y, local_z );
		if(!check) printf( "Problems with normcrossX \n");

		*(local_xyz + nsdsq*k)     = *(local_x);
		*(local_xyz + nsdsq*k + 1) = *(local_x + 1);
		*(local_xyz + nsdsq*k + 2) = *(local_x + 2);
		*(local_xyz + nsdsq*k + 3) = *(local_y);
		*(local_xyz + nsdsq*k + 4) = *(local_y + 1);
		*(local_xyz + nsdsq*k + 5) = *(local_y + 2);
		*(local_xyz + nsdsq*k + 6) = *(local_z);
		*(local_xyz + nsdsq*k + 7) = *(local_z + 1);
		*(local_xyz + nsdsq*k + 8) = *(local_z + 2);

	}

	return 1;
}


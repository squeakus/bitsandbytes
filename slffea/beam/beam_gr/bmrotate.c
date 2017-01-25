/*
    This utility function rotates vectors for a finite 
    element program which does analysis on a beam

		 Updated 2/1/00

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ONE    0.99999


int matX(double *, double *, double *, int, int, int);

int bmrotate( double *coord_el, double *vec_in, double *vec_out)
{
	int i, i1, j, k, sof, check;
	double L, Lx, Ly, Lz, Lsq, Lxysq, rotate[9];

	sof = sizeof(double);

	Lx = *(coord_el+3) - *(coord_el);
	Ly = *(coord_el+4) - *(coord_el+1);
	Lz = *(coord_el+5) - *(coord_el+2);

	/*printf(" Lx, Ly, Lz %f %f %f\n ", Lx, Ly, Lz);*/

	Lsq = Lx*Lx+Ly*Ly+Lz*Lz;
	L = sqrt(Lsq);
	Lx /= L; Ly /= L; Lz /= L;
	Lxysq = Lx*Lx + Ly*Ly;
	Lxysq = sqrt(Lxysq);

/* Assembly of the 3X3 rotation matrix matrix */

	memset(rotate,0,9*sof);
	*(rotate) = Lx;
	*(rotate+1) = Ly;
	*(rotate+2) = Lz;
	*(rotate+3) = -Ly/Lxysq;
	*(rotate+4) = Lx/Lxysq;
	*(rotate+5) = 0.0;
	*(rotate+6) = -Lx*Lz/Lxysq;
	*(rotate+7) = -Ly*Lz/Lxysq;
	*(rotate+8) = Lxysq;

/* If local x axis lies on global z axis, modify rotate */

	if(Lz > ONE)
	{
		memset(rotate,0,9*sof);
		*(rotate+2) = 1.0;
		*(rotate+4) = 1.0;
		*(rotate+6) = -1.0;
	}

	if(Lz < -ONE)
	{
		memset(rotate,0,9*sof);
		*(rotate+2) = -1.0;
		*(rotate+4) = 1.0;
		*(rotate+6) = 1.0;
	}
	check = matX(vec_out, rotate, vec_in, 3, 1, 3);

	return 1;
}


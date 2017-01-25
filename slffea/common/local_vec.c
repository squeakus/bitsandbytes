/*
    This utility function calculates the local x, y, z (called local_xyz[])
    basis vectors for each element for a finite element program
    which does analysis on a 4 node plate or quad element.

    For 3-D quad meshes, I have to rotate from the global coordinates to
    the local x and y coordinates which lie in the plane of the element.
    To do this I have to calculate the normal to the plate face, then
    cross product that normal with an in plane vector to get the other
    local axis direction.

    I also use the algorithm in my shell code that tries to align this
    normal vector as closely as possible to the global z direction.

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
#define npel4           4                          /* nodes per quad element */

extern int numel;

int normcrossX(double *, double *, double *);

int local_vectors( int *connect, double *coord, double *local_xyz ) 
{
/*
   Calculation of the basis is taken from:

       Kohneke, Peter, *ANSYS User's Manual for Revision 5.0,
	   Vol IV Theory, Swanson Analysis Systems Inc., Houston, 1992.

   page 14-152.  The basis is created by first taking the cross product of the 2 vectors
   which make up the diagonals of the plate.  This will form the local z axis.  From
   there, I use the same algorithm in Hughes to get the local x and local y.

		Updated 11/6/06
*/

	int i, i1, i2, i3, i4, i5, j, k, sof;
	int check, counter, node;
	double fdum1, fdum2, fdum3, fdum4;
	double coord_el_trans[npel4*nsd];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd];
	double xp[npel4], yp[npel4], zp[npel4];

	sof = sizeof(double);

	for( k = 0; k < numel; ++k )
	{
		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(coord_el_trans + j) = *(coord + nsd*node);
			*(coord_el_trans + npel4*1 + j) = *(coord + nsd*node + 1);
			*(coord_el_trans + npel4*2 + j) = *(coord + nsd*node + 2);
		}

		*(xp+0) = *(coord_el_trans);
		*(xp+1) = *(coord_el_trans + 1);
		*(xp+2) = *(coord_el_trans + 2);
		*(xp+3) = *(coord_el_trans + 3);

		*(yp+0) = *(coord_el_trans + npel4*1);
		*(yp+1) = *(coord_el_trans + npel4*1 + 1);
		*(yp+2) = *(coord_el_trans + npel4*1 + 2);
		*(yp+3) = *(coord_el_trans + npel4*1 + 3);

		*(zp+0) = *(coord_el_trans + npel4*2);
		*(zp+1) = *(coord_el_trans + npel4*2 + 1);
		*(zp+2) = *(coord_el_trans + npel4*2 + 2);
		*(zp+3) = *(coord_el_trans + npel4*2 + 3);

		*(vec_dum1)     = *(xp+1) - *(xp+3);
		*(vec_dum1 + 1) = *(yp+1) - *(yp+3);
		*(vec_dum1 + 2) = *(zp+1) - *(zp+3);
		*(vec_dum2)     = *(xp+2) - *(xp+0);
		*(vec_dum2 + 1) = *(yp+2) - *(yp+0);
		*(vec_dum2 + 2) = *(zp+2) - *(zp+0);

/* Calculate the local z basis vector for element k */
		check = normcrossX( vec_dum1, vec_dum2, local_z );
		if(!check) printf( "Problems with normcrossX \n");

		fdum1 = fabs(*(local_z));
		fdum2 = fabs(*(local_z+1));
		fdum3 = fabs(*(local_z+2));
/*
   The algorithm below is taken from "The Finite Element Method" by Thomas Hughes,
   page 388.  The goal is to find the local shell coordinates which come closest
   to the global x, y, z coordinates.  In the algorithm below, vec_dum is set to either
   the global x, y, or z basis vector based on the one of the 2 smaller components of the
   local z basis vector.  Once set, the cross product of the local z vector and vec_dum
   produces the local y direction.  This local y is then crossed with the local z to
   produce local x.
*/
		memset(vec_dum,0,nsd*sof);
		i2=1;
		if( fdum1 > fdum3)
		{
			fdum3=fdum1;
			i2=2;
		}
		if( fdum2 > fdum3) i2=3;
		*(vec_dum+(i2-1))=1.0;

/* Calculate the local y basis vector */
		check = normcrossX( local_z, vec_dum, local_y );
		if(!check) printf( "Problems with normcrossX \n");

/* Calculate the local x basis vector */
		check = normcrossX( local_y, local_z, local_x );
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

int local_vectors_old( int *connect, double *coord, double *local_xyz ) 
{
/*
   Below is the original method I used to create the basis.  I created a local basis for
   every node of every element based on the cross product of the sides connected to a node
   to create the local z axis of that node.  This is how it is done in the doubly curved shell,
   although the shell has its own local vectors function.

   By having multiple basses, I was hoping not to bias the rotation to a particular node. 
   Each submatrix of the full rotation matrix is made up of the rotation vectors generated by
   that particular node.  The code for this is below.

		Updated 11/6/06
*/
	int i, i1, i2, i3, i4, i5, j, k, sof;
	int check, counter, node;
	double fdum1, fdum2, fdum3, fdum4;
	double coord_el_trans[npel4*nsd];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd];
	double xp[npel4], xp0[npel4], xp1[npel4], xp2[npel4],
		yp[npel4], yp0[npel4], yp1[npel4], yp2[npel4],
		zp[npel4], zp0[npel4], zp1[npel4], zp2[npel4];

	sof = sizeof(double);

	for( k = 0; k < numel; ++k )
	{

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(coord_el_trans + j) = *(coord + nsd*node);
			*(coord_el_trans + npel4*1 + j) = *(coord + nsd*node + 1);
			*(coord_el_trans + npel4*2 + j) = *(coord + nsd*node + 2);
		}

		*(xp2+1) = *(xp1+3) = *(xp0+0) = *(coord_el_trans);
		*(xp2+2) = *(xp1+0) = *(xp0+1) = *(coord_el_trans + 1);
		*(xp2+3) = *(xp1+1) = *(xp0+2) = *(coord_el_trans + 2);
		*(xp2+0) = *(xp1+2) = *(xp0+3) = *(coord_el_trans + 3);

		*(yp2+1) = *(yp1+3) = *(yp0+0) = *(coord_el_trans + npel4*1);
		*(yp2+2) = *(yp1+0) = *(yp0+1) = *(coord_el_trans + npel4*1 + 1);
		*(yp2+3) = *(yp1+1) = *(yp0+2) = *(coord_el_trans + npel4*1 + 2);
		*(yp2+0) = *(yp1+2) = *(yp0+3) = *(coord_el_trans + npel4*1 + 3);

		*(zp2+1) = *(zp1+3) = *(zp0+0) = *(coord_el_trans + npel4*2);
		*(zp2+2) = *(zp1+0) = *(zp0+1) = *(coord_el_trans + npel4*2 + 1);
		*(zp2+3) = *(zp1+1) = *(zp0+2) = *(coord_el_trans + npel4*2 + 2);
		*(zp2+0) = *(zp1+2) = *(zp0+3) = *(coord_el_trans + npel4*2 + 3);

		for( j = 0; j < npel4; ++j )
		{
			*(vec_dum1)     = *(xp1+j) - *(xp0+j);
			*(vec_dum1 + 1) = *(yp1+j) - *(yp0+j);
			*(vec_dum1 + 2) = *(zp1+j) - *(zp0+j);
			*(vec_dum2)     = *(xp2+j) - *(xp0+j);
			*(vec_dum2 + 1) = *(yp2+j) - *(yp0+j);
			*(vec_dum2 + 2) = *(zp2+j) - *(zp0+j);

/* Calculate the local z basis vector for element k, node j */
			check = normcrossX( vec_dum1, vec_dum2, local_z );
			if(!check) printf( "Problems with normcrossX \n");

			fdum1 = fabs(*(local_z));
			fdum2 = fabs(*(local_z+1));
			fdum3 = fabs(*(local_z+2));
/*
   The algorithm below is taken from "The Finite Element Method" by Thomas Hughes,
   page 388.  The goal is to find the local shell coordinates which come closest
   to the global x, y, z coordinates.  In the algorithm below, vec_dum is set to either
   the global x, y, or z basis vector based on the one of the 2 smaller components of the
   local z basis vector.  Once set, the cross product of the local z vector and vec_dum
   produces the local y direction.  This local y is then crossed with the local z to
   produce local x.
*/
			memset(vec_dum,0,nsd*sof);
			i2=1;
			if( fdum1 > fdum3)
			{
			    fdum3=fdum1;
			    i2=2;
			}
			if( fdum2 > fdum3) i2=3;
			*(vec_dum+(i2-1))=1.0;

/* Calculate the local y basis vector for node j */
			check = normcrossX( local_z, vec_dum, local_y );
			if(!check) printf( "Problems with normcrossX \n");

/* Calculate the local x basis vector for node j */
			check = normcrossX( local_y, local_z, local_x );
			if(!check) printf( "Problems with normcrossX \n");

			*(local_xyz + npel4*nsdsq*k + nsdsq*j)     = *(local_x);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 1) = *(local_x + 1);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 2) = *(local_x + 2);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 3) = *(local_y);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 4) = *(local_y + 1);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 5) = *(local_y + 2);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 6) = *(local_z);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 7) = *(local_z + 1);
			*(local_xyz + npel4*nsdsq*k + nsdsq*j + 8) = *(local_z + 2);
		}

	}

	return 1;
}


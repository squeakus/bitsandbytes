/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a quad.  It is for modal analysis.

		Updated 9/8/06

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
#include "qdconst.h"
#include "qdstruct.h"

extern int numel, numnp, dof, sof, flag_3D;
extern double shg[sosh], shl[sosh], w[num_int], *Area0;
extern int consistent_mass_flag, consistent_mass_store;

int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int quadB_mass(double *,double *);

int qdshg_mass( double *, int, double *, double *);

int qdMassPassemble(int *connect, double *coord, int *el_matl, double *mass,
	MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter, dum;
	int matl_num;
	double rho, fdum, fdum1, fdum2, fdum3, fdum4;
	double B_mass[MsoB], B2_mass[MsoB];
	double M_temp[neqlsq], M_el[neqlsq], rotate[npel*nsd2*nsd];
	double U_el[neqel];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double det[num_int];
	double P_el[neqel];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd], vec_dum4[nsd];
	double xp[npel], xp0[npel], xp1[npel], xp2[npel],
		yp[npel], yp0[npel], yp1[npel], yp2[npel],
		zp[npel], zp0[npel], zp1[npel], zp2[npel];

	memset(P_global,0,dof*sof);

	memcpy(shg,shl,sosh*sizeof(double));

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (mass + k*neqlsq), U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}
	    }
	}
	else
	{

/* Assemble P matrix by re-deriving element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {
		matl_num = *(el_matl+k);
		rho = matl[matl_num].rho;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

		memset(rotate,0,npel*nsd2*nsd*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel + j) = *(coord_el_trans + npel*1 + j);
		    }
		}
		else
		{

/* For 3-D quad meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  To do this I have to calculate
   the normal to the plate face, then cross product that normal with an in plane vector
   to get the other local axis direction.

   I also use the algorithm in my shell code that tries to align this normal vector
   as closely as possible to the global z direction.
*/

/*
   I considered many different methods for calculating the initial local z axis.  For a
   complete discussion of this, read the comments given in "qdkasmbl.c".  This is a legacy
   code for the method I did not use.
*/

		    *(xp2+1) = *(xp1+3) = *(xp0+0) = *(coord_el_trans);
		    *(xp2+2) = *(xp1+0) = *(xp0+1) = *(coord_el_trans + 1);
		    *(xp2+3) = *(xp1+1) = *(xp0+2) = *(coord_el_trans + 2);
		    *(xp2+0) = *(xp1+2) = *(xp0+3) = *(coord_el_trans + 3);

		    *(yp2+1) = *(yp1+3) = *(yp0+0) = *(coord_el_trans + npel*1);
		    *(yp2+2) = *(yp1+0) = *(yp0+1) = *(coord_el_trans + npel*1 + 1);
		    *(yp2+3) = *(yp1+1) = *(yp0+2) = *(coord_el_trans + npel*1 + 2);
		    *(yp2+0) = *(yp1+2) = *(yp0+3) = *(coord_el_trans + npel*1 + 3);

		    *(zp2+1) = *(zp1+3) = *(zp0+0) = *(coord_el_trans + npel*2);
		    *(zp2+2) = *(zp1+0) = *(zp0+1) = *(coord_el_trans + npel*2 + 1);
		    *(zp2+3) = *(zp1+1) = *(zp0+2) = *(coord_el_trans + npel*2 + 2);
		    *(zp2+0) = *(zp1+2) = *(zp0+3) = *(coord_el_trans + npel*2 + 3);

		    for( j = 0; j < npel; ++j )
		    {
			*(vec_dum1)     = *(xp1+j) - *(xp0+j);
			*(vec_dum1 + 1) = *(yp1+j) - *(yp0+j);
			*(vec_dum1 + 2) = *(zp1+j) - *(zp0+j);
			*(vec_dum2)     = *(xp2+j) - *(xp0+j);
			*(vec_dum2 + 1) = *(yp2+j) - *(yp0+j);
			*(vec_dum2 + 2) = *(zp2+j) - *(zp0+j);

/* Calculate the local z basis vector for node j */
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

			*(rotate + j*nsd2*nsd) = *(local_x);
			*(rotate + j*nsd2*nsd + 1) = *(local_x + 1);
			*(rotate + j*nsd2*nsd + 2) = *(local_x + 2);
			*(rotate + j*nsd2*nsd + 3) = *(local_y);
			*(rotate + j*nsd2*nsd + 4) = *(local_y + 1);
			*(rotate + j*nsd2*nsd + 5) = *(local_y + 2);

/* Put coord_el into local coordinates */

			check = matX( (coord_el_local+nsd2*j), (rotate + j*nsd2*nsd),
				(coord_el+nsd*j), nsd2, 1, nsd);
			if(!check) printf( "Problems with  matX \n");
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
		    }

/* The code below does the same as the matrix multiplication above to calculate the local
   coordinates and is clearer but isn't as efficient.  If the code below is used, make sure
   to comment out both the matrix multiplication above as well as the setting of
   coord_el_trans.

		    dum = nsd*npel;
		    check = rotXmat3(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat3 \n");
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
		    }
*/
		}

/* The call to qdshg_mass is only for calculating the determinent */

		check = qdshg_mass(det, k, shg, coord_el_local_trans);
		if(!check) printf( "Problems with qdshg_mass \n");
#if 0
		for( i1 = 0; i1 < num_int; ++i1 )
		{
		    for( i2 = 0; i2 < npel; ++i2 )
		    {
			printf("%10.6f ",*(shl+npel*(nsd2+1)*i1 + npel*(nsd2) + i2));
		    }
		    printf(" \n ");
		}
		printf(" \n ");
#endif

/* The loop over j below calculates the 4 points of the gaussian integration
   for several quantities */

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B_mass,0,MsoB*sof);
		    memset(B2_mass,0,MsoB*sof);
		    memset(M_temp,0,neqlsq*sof);

/* Assembly of the B matrix for mass */

		    check = quadB_mass((shg+npel*(nsd2+1)*j + npel*(nsd2)),B_mass);
		    if(!check) printf( "Problems with quadB_mass \n");

#if 0
		    for( i1 = 0; i1 < nsd2; ++i1 )
		    {
			for( i2 = 0; i2 < neqel8; ++i2 )
			{
				printf("%9.6f ",*(B_mass+neqel8*i1+i2));
			}
			printf(" \n ");
		    }
		    printf(" \n ");
#endif
		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

		    check=matXT(M_temp, B_mass, B2_mass, neqel8, neqel8, nsd2);
		    if(!check) printf( "Problems with matXT \n");

		    fdum = rho*(*(w+j))*(*(det+j));
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(M_el+i2) += *(M_temp+i2)*fdum;
		    }
		}

		if(!flag_3D)
		{
/* For 2-D meshes */
		   for( i = 0; i < npel; ++i )
		   {
			for( j = 0; j < npel; ++j )
			{
/* row for displacement x */
			   *(M_temp + ndof*neqel*i + ndof*j) =
				*(M_el + ndof2*neqel8*i + ndof2*j);
			   *(M_temp + ndof*neqel*i + ndof*j + 1) =
				*(M_el + ndof2*neqel8*i + ndof2*j + 1);
			   *(M_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(M_temp + ndof*neqel*i + neqel  + ndof*j) =
				*(M_el + ndof2*neqel8*i + neqel8 + ndof2*j);
			   *(M_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(M_el + ndof2*neqel8*i + neqel8 + ndof2*j + 1);
			   *(M_temp + ndof*neqel*i + neqel + ndof*j + 2) = 0.0;

/* row for displacement z */
			   *(M_temp + ndof*neqel*i + 2*neqel + ndof*j) = 0.0;
			   *(M_temp + ndof*neqel*i + 2*neqel + ndof*j + 1) = 0.0;
			   *(M_temp + ndof*neqel*i + 2*neqel + ndof*j + 2) = 0.0;
			}
		   }
		   memcpy(M_el, M_temp, neqlsq*sizeof(double));
		}
		else
		{
/* For 3-D meshes */

/* Put M back to global coordinates */

		   check = matXrot2(M_temp, M_el, rotate, neqel8, neqel);
		   if(!check) printf( "Problems with matXrot2 \n");

		   check = rotTXmat2(M_el, rotate, M_temp, neqel, neqel);
		   if(!check) printf( "Problems with rotTXmat2 \n");
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, M_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}

	    }
	}

	return 1;
}

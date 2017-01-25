/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a triangle.  It is for modal analysis.

		Updated 10/20/06

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
#include "trconst.h"
#include "trstruct.h"

extern int numel, numnp, dof, sof, flag_3D;
extern double shg[sosh], shl[sosh], w[num_int], *Area0;
extern int consistent_mass_flag, consistent_mass_store;

int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matXT(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int triangleB_mass(double *,double *);

int trMassPassemble(int *connect, double *coord, int *el_matl, double *local_xyz,
	double *mass, MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter, dum;
	int matl_num;
	double rho, thickness, fdum;
	double B_mass[MsoB], B2_mass[MsoB];
	double M_temp[neqlsq], M_el[neqlsq], rotate[nsd2*nsd];
	double U_el[neqel];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3, X12, X13, Y12, Y13, Z12, Z13;
	double Xp, Xr, Xq, Yp, Yr, Yq, Zp, Zr, Zq, Xpr, Xpq, Ypr, Ypq, Zpr, Zpq;
	double Lpq, Ltr, Mpq, Mtr, Npq, Ntr, Dpq, Dpt, Dtr;
	double det[num_int], area_el, wXdet;
	double P_el[neqel];

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
		thickness = matl[matl_num].thick;
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

		memset(rotate,0,nsd2*nsd*sof);

/* Assembly of the Mass matrix.
*/

		if(!flag_3D)
		{
/* For 2-D meshes */
		   X1 = *(coord_el_trans);
		   X2 = *(coord_el_trans + 1);
		   X3 = *(coord_el_trans + 2);

		   Y1 = *(coord_el_trans + npel*1);
		   Y2 = *(coord_el_trans + npel*1 + 1);
		   Y3 = *(coord_el_trans + npel*1 + 2);
		}
		else
		{
/* For 3-D meshes */

/* For 3-D triangle meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  The local basis used for the
   rotation has already been calculated and stored in local_xyz[].  Below, it is
   copied to rotate[].
*/
		   *(rotate)     = *(local_xyz + nsdsq*k);
		   *(rotate + 1) = *(local_xyz + nsdsq*k + 1);
		   *(rotate + 2) = *(local_xyz + nsdsq*k + 2);
		   *(rotate + 3) = *(local_xyz + nsdsq*k + 3);
		   *(rotate + 4) = *(local_xyz + nsdsq*k + 4);
		   *(rotate + 5) = *(local_xyz + nsdsq*k + 5);

/* Put coord_el into local coordinates */

		   dum = nsd*npel;
		   check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		   if(!check) printf( "Problems with  rotXmat2 \n");
/*
     It should be noted that I have permutated the node sequences.
     (see Hughes Figure 3.I.5 on page 167; node 3 in Hughes is node 1
     in SLFFEA, node 2 is node 3, and node 1 goes to node 2.)
     This is because I want to be consistant with the tetrahedron.  You can
     read more about this change in teshl for the tetrahedron.

     This change only effects the calculation of the area.  The [M_el] matrix
     is still the same.
*/
		   X1 = *(coord_el_local);     Y1 = *(coord_el_local + 1);
		   X2 = *(coord_el_local + 2); Y2 = *(coord_el_local + 3);
		   X3 = *(coord_el_local + 4); Y3 = *(coord_el_local + 5);
		}

		*(coord_el_local_trans) = X1;     *(coord_el_local_trans + 3) = Y1;
		*(coord_el_local_trans + 1) = X2; *(coord_el_local_trans + 4) = Y2;
		*(coord_el_local_trans + 2) = X3; *(coord_el_local_trans + 5) = Y3;

/* Area is simply the the cross product of the sides connecting node 1 to node 0
   and node 2 to node 0, divided by 2.
*/

		fdum = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 2*Area:%f\n", k, fdum);
			return 0;
		}

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		area_el = .5*fdum;

/* Normally, with plane elements, we assume a unit thickness in the transverse direction.  But
   because these elements can be 3 dimensional, I multiply the material property matrix by the
   thickness.  This is justified by equation 5.141a in "Theory of Matrix Structural
   Analysis" by J. S. Przemieniecki on page 86.
*/

		fdum = area_el*rho*thickness/12.0;

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq*sof);
/*
   This is taken from "Theory of Matrix Structural Analysis" by
   J. S. Przemieniecki, page 297-299.
*/

		*(M_el)    = 2.0*fdum; *(M_el+2)  = 1.0*fdum; *(M_el+4)  = 1.0*fdum;
		*(M_el+7)  = 2.0*fdum; *(M_el+9)  = 1.0*fdum; *(M_el+11) = 1.0*fdum;
		*(M_el+12) = 1.0*fdum; *(M_el+14) = 2.0*fdum; *(M_el+16) = 1.0*fdum;
		*(M_el+19) = 1.0*fdum; *(M_el+21) = 2.0*fdum; *(M_el+23) = 1.0*fdum;
		*(M_el+24) = 1.0*fdum; *(M_el+26) = 1.0*fdum; *(M_el+28) = 2.0*fdum;
		*(M_el+31) = 1.0*fdum; *(M_el+33) = 1.0*fdum; *(M_el+35) = 2.0*fdum;

		memset(M_temp,0,neqlsq*sof);

		if(!flag_3D)
		{
/* For 2-D meshes */
		   for( i = 0; i < npel; ++i )
		   {
			for( j = 0; j < npel; ++j )
			{
/* row for displacement x */
			   *(M_temp + ndof*neqel*i + ndof*j) =
				*(M_el + ndof2*neqel6*i + ndof2*j);
			   *(M_temp + ndof*neqel*i + ndof*j + 1) =
				*(M_el + ndof2*neqel6*i + ndof2*j + 1);
			   *(M_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(M_temp + ndof*neqel*i + neqel  + ndof*j) =
				*(M_el + ndof2*neqel6*i + neqel6 + ndof2*j);
			   *(M_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(M_el + ndof2*neqel6*i + neqel6 + ndof2*j + 1);
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

		   check = matXrot2(M_temp, M_el, rotate, neqel6, neqel);
		   if(!check) printf( "Problems with matXrot2 \n");

		   check = rotTXmat2(M_el, rotate, M_temp, neqel, neqel);
		   if(!check) printf( "Problems with rotTXmat2 \n");
		}
/*
		for( i1 = 0; i1 < neqel; ++i1 )
		{
			for( i2 = 0; i2 < neqel; ++i2 )
			{
				printf("%16.8e ",*(M_el+neqel*i1+i2));
			}
			printf(" \n");
		}
		printf(" \n");
*/

		/* printf("This is 2 X Area %10.6f for element %4d\n",2.0*area_el,k);
		printf("%4i %16.8e\n",k, area_el);*/

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

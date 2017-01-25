/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a beam.  It is for modal analysis.

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
#include "bmconst.h"
#include "bmstruct.h"
#include "bmshape_struct.h"

extern int numel, numnp, dof, sof;
extern int consistent_mass_flag, consistent_mass_store;

int bmnormcrossX(double *, double *, double *);

int matX(double *, double *, double *, int, int, int);

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int bmMassPassemble( int *connect, double *coord, int *el_matl, int *el_type,
	double *length, double *local_xyz, double *mass, MATL *matl,
	double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, ij, dof_el[neqel];
	int check, counter, dum, node0, node1;
	int matl_num, el_num, type_num;
	double area, rho, Iydivarea, Izdivarea, Ipdivarea, fdum, fdum2;
	double L, Lsq;
	double M_temp[neqlsq], M_el[neqlsq], rotate[nsdsq];
	double U_el[neqel];
	double jacob;
	double P_el[neqel];
	SHAPE sh;

	memset(P_global,0,dof*sof);

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		*(dof_el) = ndof*node0;
		*(dof_el+1) = ndof*node0+1;
		*(dof_el+2) = ndof*node0+2;
		*(dof_el+3) = ndof*node0+3;
		*(dof_el+4) = ndof*node0+4;
		*(dof_el+5) = ndof*node0+5;

		*(dof_el+6) = ndof*node1;
		*(dof_el+7) = ndof*node1+1;
		*(dof_el+8) = ndof*node1+2;
		*(dof_el+9) = ndof*node1+3;
		*(dof_el+10) = ndof*node1+4;
		*(dof_el+11) = ndof*node1+5;

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
		type_num = *(el_type+k);
		rho = matl[matl_num].rho;
		area = matl[matl_num].area;
		Iydivarea = matl[matl_num].Iy/area;
		Izdivarea = matl[matl_num].Iz/area;
		Ipdivarea = (matl[matl_num].Iy + matl[matl_num].Iz)/area;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		L = *(length + k);
		Lsq = L*L;

		jacob = L/2.0;

		memset(rotate,0,nsdsq*sof);

		*(rotate)     = *(local_xyz + nsdsq*k);
		*(rotate + 1) = *(local_xyz + nsdsq*k + 1);
		*(rotate + 2) = *(local_xyz + nsdsq*k + 2);
		*(rotate + 3) = *(local_xyz + nsdsq*k + 3);
		*(rotate + 4) = *(local_xyz + nsdsq*k + 4);
		*(rotate + 5) = *(local_xyz + nsdsq*k + 5);
		*(rotate + 6) = *(local_xyz + nsdsq*k + 6);
		*(rotate + 7) = *(local_xyz + nsdsq*k + 7);
		*(rotate + 8) = *(local_xyz + nsdsq*k + 8);

/* defining the components of a el element vector */

		*(dof_el) = ndof*node0;
		*(dof_el+1) = ndof*node0+1;
		*(dof_el+2) = ndof*node0+2;
		*(dof_el+3) = ndof*node0+3;
		*(dof_el+4) = ndof*node0+4;
		*(dof_el+5) = ndof*node0+5;

		*(dof_el+6) = ndof*node1;
		*(dof_el+7) = ndof*node1+1;
		*(dof_el+8) = ndof*node1+2;
		*(dof_el+9) = ndof*node1+3;
		*(dof_el+10) = ndof*node1+4;
		*(dof_el+11) = ndof*node1+5;

		memset(M_el,0,neqlsq*sof);
		memset(M_temp,0,neqlsq*sof);

/* Assembly of the local mass matrix:

   For a truss, the only non-zero components are those for the axial DOF terms.  So
   leave as zero in [M_el] everything except *(M_el) and *(M_el+6)
   and *(M_el+72) and *(M_el+78) if the type_num = 1.

   Normally, we would form the product of [mass_el] = rho*[B_mass(transpose)][B_mass],
   but this cannot be done for beams because certain terms integrate to zero
   by analytical inspection, and so this matrix must be explicitly coded.  The zero
   terms pertain to integrations involving center of inertia, product of inertia, etc.
   The reason this can be done for the stiffness matrix is because the [B] matrix
   has a form which maintains the zeros of [K_el].

   Instead, I will use the fully integrated mass matrix as given by 
   "Theory of Matrix Structural Analysis" by J. S. Przemieniecki on page 295.
   
*/
/* row 0 */
		*(M_el) = 1.0/3.0;
		*(M_el+6) = 1.0/6.0;
		*(M_el+72) = *(M_el+6);
/* row 6 */
		*(M_el+78) = 1.0/3.0;

		if(type_num < 2)
		{
/* For truss element */

/* row 1 */
		    *(M_el+13) = 1.0/3.0;
		    *(M_el+19) = 1.0/6.0;
		    *(M_el+85) = *(M_el+19);
/* row 2 */
		    *(M_el+26) = 1.0/3.0;
		    *(M_el+32) = 1.0/6.0;
		    *(M_el+98) = *(M_el+32);
/* row 7 */
		    *(M_el+91) = 1.0/3.0;
/* row 8 */
		    *(M_el+104) = 1.0/3.0;
		}

		if(type_num > 1)
		{
/* row 1 */
			*(M_el+13) = 13.0/35.0 + 6.0*Izdivarea/(5.0*Lsq);

			*(M_el+17) = 11.0*L/210.0 + Izdivarea/(10.0*L);
			*(M_el+61) = *(M_el+17);

			*(M_el+19) = 9.0/70.0 - 6.0*Izdivarea/(5.0*Lsq);
			*(M_el+85) = *(M_el+19);

			*(M_el+23) =  -13.0*L/420.0 + Izdivarea/(10.0*L);
			*(M_el+133) = *(M_el+23);
/* row 2 */
			*(M_el+26) = 13.0/35.0 + 6.0*Iydivarea/(5.0*Lsq);

			*(M_el+28) = -11.0*L/210.0 - Iydivarea/(10.0*L);
			*(M_el+50) = *(M_el+28);

			*(M_el+32) = 9.0/70.0 - 6.0*Iydivarea/(5.0*Lsq);
			*(M_el+98) = *(M_el+32);

			*(M_el+34) = 13.0*L/420.0 - Iydivarea/(10.0*L);
			*(M_el+122) = *(M_el+34);
/* row 3 */
			*(M_el+39) = Ipdivarea/3.0;

			*(M_el+45) = Ipdivarea/6.0;
			*(M_el+111) = *(M_el+45);
/* row 4 */
			*(M_el+52) = Lsq/105.0 + 2.0*Iydivarea/15.0;

			*(M_el+56) = -13.0*L/420.0 + Iydivarea/(10.0*L);
			*(M_el+100) = *(M_el+56);

			*(M_el+58) = -Lsq/140.0 - Iydivarea/30.0;
			*(M_el+124) = *(M_el+58);
/* row 5 */
			*(M_el+65) = Lsq/105.0 + 2.0*Izdivarea/15.0;

			*(M_el+67) = 13.0*L/420.0 - Izdivarea/(10.0*L);
			*(M_el+89) = *(M_el+67);

			*(M_el+71) = -Lsq/140.0 - Izdivarea/30.0;
			*(M_el+137) = *(M_el+71);
/* row 7 */
			*(M_el+91) = 13.0/35.0 + 6.0*Izdivarea/(5.0*Lsq);

			*(M_el+95) = -11.0*L/210.0 - Izdivarea/(10.0*L);
			*(M_el+139) = *(M_el+95);
/* row 8 */
			*(M_el+104) = 13.0/35.0 + 6.0*Iydivarea/(5.0*Lsq);

			*(M_el+106) = 11.0*L/210.0 - Iydivarea/(10.0*L);
			*(M_el+128) = *(M_el+106);
/* row 9 */
			*(M_el+117) = Ipdivarea/3.0;
/* row 10 */
			*(M_el+130) =  Lsq/105.0 + 2.0*Iydivarea/15.0;
/* row 11 */
			*(M_el+143) =  Lsq/105.0 + 2.0*Izdivarea/15.0;
		}

		fdum = rho*area*L;
		for( i1 = 0; i1 < neqlsq; ++i1 )
		{
			*(M_el + i1) = *(M_el + i1)*fdum;
		}

/* Put M_el back to global coordinates */

		check = matXrot(M_temp, M_el, rotate, neqel, neqel);
		if(!check) printf( "Problems with matXrot \n");

		check = rotTXmat(M_el, rotate, M_temp, neqel, neqel);
		if(!check) printf( "Problems with rotTXmat \n");

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

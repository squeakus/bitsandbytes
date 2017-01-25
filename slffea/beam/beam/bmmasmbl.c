/*
    This utility function assembles the lumped sum Mass matrix for a
    finite element program which does analysis on a beam.  It is for
    modal analysis.

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

extern int dof, numel, numnp, sof;
extern int consistent_mass_flag, consistent_mass_store, lumped_mass_flag;
extern double x3[num_int3], w3[num_int3], x4[num_int4], w4[num_int4];

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int bmshape_mass(SHAPE *, double , double , double );

int bmnormcrossX(double *, double *, double *);

int bmMassemble( int *connect, double *coord, int *el_matl, int *el_type, int *id,
	double *length, double *local_xyz, double *mass, MATL *matl)
{
	int i, i1, i2, i3, j, k, ij, dof_el[neqel];
	int check, counter, dum, node0, node1;
	int matl_num, el_num, type_num;
	double area, rho, Iydivarea, Izdivarea, Ipdivarea, fdum, fdum2;
	double L, Lsq;
	double M_temp[neqlsq], M_el[neqlsq], mass_el[neqel],
		mass_local[neqlsq], rotate[nsdsq];
	double jacob;
	SHAPE sh;

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

		/*printf(" Lx, Ly, Lz %f %f %f\n ", Lx, Ly, Lz);*/

		L = *(length + k);
		Lsq = L*L;

		if( consistent_mass_flag )
		{

/* Only need to calculate rotations for consistent masses */

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
		}

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
		memset(mass_el,0,neqel*sof);

		fdum = .5*rho*area*L;
		if( lumped_mass_flag )
		{
/* The lumped mass matrix is taken from the "ANSYS User's Manual for Revision 5.0",
   edited by Kohneke, Peter, page 14-12, eq. 14.4-3.
*/
		    *(M_el) = fdum;
		    *(M_el+6) = fdum;
		    *(M_el+1) = fdum;
		    *(M_el+2) = fdum;
		    *(M_el+3) = 0.0;
		    *(M_el+4) = 0.0;
		    *(M_el+5) = 0.0;
		    *(M_el+7) = fdum;
		    *(M_el+8) = fdum;
		    *(M_el+9) = 0.0;
		    *(M_el+10) = 0.0;
		    *(M_el+11) = 0.0;
		}

#if 1
		if( consistent_mass_flag )
		{

/* Assembly of the local mass matrix:

   For a truss, the only non-zero components are those for the axial DOF terms.  So
   leave as zero in [mass_local] everything except *(mass_local) and *(mass_local+6)
   and *(mass_local+72) and *(mass_local+78) if the type_num = 1.

   Normally, we would form the product of [mass_local] = rho*[B_mass(transpose)][B_mass],
   but this cannot be done for beams because certain terms integrate to zero
   by analytical inspection, and so this matrix must be explicitly coded.  The zero
   terms pertain to integrations involving center of inertia, product of inertia, etc.
   The reason this can be done for the stiffness matrix is because the [B] matrix
   has a form which maintains the zeros of [K_el].

   Instead, I will use the fully integrated mass matrix as given by 
   "Theory of Matrix Structural Analysis" by J. S. Przemieniecki on page 294.
   
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
		}

#endif

#if 0
		if( consistent_mass_flag )
		{

/* Assembly of the local mass matrix:

   Below, I am assembling the above mass matrix based on numerical integration. 
   The reasons I am doing this are 

      1) To help debug the above code
      2) To illustrate 4 point Gauss integration.

   Because it is less efficient than simply using a pre-integrated matrix, it has
   been commented out. 
*/

/* The loop below calculates the 4 points of the gaussian integration */

		    for( j = 0; j < num_int_mass; ++j )
		    {
			memset(mass_local,0,neqlsq*sof);

			check = bmshape_mass(&sh, *(x4+j), L, Lsq);
			if(!check) printf( "Problems with bmshape \n");
/* row 0 */
			*(mass_local) = sh.Nhat[0].dx0*sh.Nhat[0].dx0;
			*(mass_local+6) = sh.Nhat[0].dx0*sh.Nhat[1].dx0;
			*(mass_local+72) = *(mass_local+6);
/* row 6 */
			*(mass_local+78) = sh.Nhat[1].dx0*sh.Nhat[1].dx0;

			if(type_num < 2)
			{
/* For truss element */

/* row 1 */
			    *(mass_local+13) = sh.Nhat[0].dx0*sh.Nhat[0].dx0;
			    *(mass_local+19) = sh.Nhat[0].dx0*sh.Nhat[1].dx0;
			    *(mass_local+85) = *(mass_local+19);
/* row 2 */
			    *(mass_local+26) = sh.Nhat[0].dx0*sh.Nhat[0].dx0;
			    *(mass_local+32) = sh.Nhat[0].dx0*sh.Nhat[1].dx0;
			    *(mass_local+98) = *(mass_local+32);
/* row 7 */
			    *(mass_local+91) = sh.Nhat[0].dx0*sh.Nhat[0].dx0;
/* row 8 */
			    *(mass_local+104) = sh.Nhat[0].dx0*sh.Nhat[0].dx0;
			}

			if(type_num > 1)
			{
/* row 1 */
			    *(mass_local+13) = sh.N[2].dx1*sh.N[2].dx1*Izdivarea +
				sh.N[0].dx0*sh.N[0].dx0;

			    *(mass_local+17) = -sh.N[1].dx1*sh.N[2].dx1*Izdivarea +
				sh.N[0].dx0*sh.N[1].dx0;
			    *(mass_local+61) = *(mass_local+17);

			    *(mass_local+19) = sh.N[0].dx1*sh.N[2].dx1*Izdivarea +
				sh.N[0].dx0*sh.N[2].dx0;
			    *(mass_local+85) = *(mass_local+19);

			    *(mass_local+23) = -sh.N[2].dx1*sh.N[3].dx1*Izdivarea +
				sh.N[0].dx0*sh.N[3].dx0;
			    *(mass_local+133) = *(mass_local+23);
/* row 2 */
			    *(mass_local+26) = sh.N[2].dx1*sh.N[2].dx1*Iydivarea +
				sh.N[0].dx0*sh.N[0].dx0;

			    *(mass_local+28) = sh.N[1].dx1*sh.N[2].dx1*Iydivarea -
				sh.N[0].dx0*sh.N[1].dx0;
			    *(mass_local+50) = *(mass_local+28);

			    *(mass_local+32) = sh.N[0].dx1*sh.N[2].dx1*Iydivarea +
				sh.N[0].dx0*sh.N[2].dx0;
			    *(mass_local+98) = *(mass_local+32);

			    *(mass_local+34) = sh.N[2].dx1*sh.N[3].dx1*Iydivarea -
				sh.N[0].dx0*sh.N[3].dx0;
			    *(mass_local+122) = *(mass_local+34);
/* row 3 */
			    *(mass_local+39) = sh.Nhat[0].dx0*sh.Nhat[0].dx0*Ipdivarea;

			    *(mass_local+45) = sh.Nhat[0].dx0*sh.Nhat[1].dx0*Ipdivarea;
			    *(mass_local+111) = *(mass_local+45);
/* row 4 */
			    *(mass_local+52) = sh.N[1].dx1*sh.N[1].dx1*Iydivarea +
				sh.N[1].dx0*sh.N[1].dx0;

			    *(mass_local+56) = sh.N[0].dx1*sh.N[1].dx1*Iydivarea -
				sh.N[1].dx0*sh.N[2].dx0;
			    *(mass_local+100) = *(mass_local+56);

			    *(mass_local+58) = sh.N[1].dx1*sh.N[3].dx1*Iydivarea +
				sh.N[1].dx0*sh.N[3].dx0;
			    *(mass_local+124) = *(mass_local+58);
/* row 5 */
			    *(mass_local+65) = sh.N[1].dx1*sh.N[1].dx1*Izdivarea +
				sh.N[1].dx0*sh.N[1].dx0;

			    *(mass_local+67) = -sh.N[0].dx1*sh.N[1].dx1*Izdivarea +
				sh.N[1].dx0*sh.N[2].dx0;
			    *(mass_local+89) = *(mass_local+67);

			    *(mass_local+71) = sh.N[1].dx1*sh.N[3].dx1*Izdivarea +
				sh.N[1].dx0*sh.N[3].dx0;
			    *(mass_local+137) = *(mass_local+71);
/* row 7 */
			    *(mass_local+91) = sh.N[0].dx1*sh.N[0].dx1*Izdivarea +
				sh.N[2].dx0*sh.N[2].dx0;

			    *(mass_local+95) = -sh.N[0].dx1*sh.N[3].dx1*Izdivarea +
				sh.N[2].dx0*sh.N[3].dx0;
			    *(mass_local+139) = *(mass_local+95);
/* row 8 */
			    *(mass_local+104) = sh.N[0].dx1*sh.N[0].dx1*Iydivarea +
				sh.N[2].dx0*sh.N[2].dx0;

			    *(mass_local+106) = -sh.N[0].dx1*sh.N[3].dx1*Iydivarea -
				sh.N[2].dx0*sh.N[3].dx0;
			    *(mass_local+128) = *(mass_local+106);
/* row 9 */
			    *(mass_local+117) = sh.Nhat[0].dx0*sh.Nhat[0].dx0*Ipdivarea;
/* row 10 */
			    *(mass_local+130) = sh.Nhat[3].dx1*sh.Nhat[3].dx1*Iydivarea +
				sh.N[3].dx0*sh.N[3].dx0;
/* row 11 */
			    *(mass_local+143) = sh.Nhat[3].dx1*sh.Nhat[3].dx1*Izdivarea +
				sh.N[3].dx0*sh.N[3].dx0;

			}

			fdum = rho*area*(*(w4+j))*jacob;
			for( i1 = 0; i1 < neqlsq; ++i1 )
			{
			    *(M_el + i1) += *(mass_local + i1)*fdum;
			}
			/* *(M_el) = 1.0/3.0*rho*area*L;*/

		    }
		}
#endif

		if(lumped_mass_flag)
		{

/* Creating the diagonal lumped mass Matrix */

		    for( j = 0; j < neqel; ++j )
		    {
			*(mass+*(dof_el+j)) += *(M_el + j);
		    }
		}

		if(consistent_mass_flag)
		{

/* Put M_el back to global coordinates */

		    check = matXrot(M_temp, M_el, rotate, neqel, neqel);
		    if(!check) printf( "Problems with matXrot \n");

		    check = rotTXmat(M_el, rotate, M_temp, neqel, neqel);
		    if(!check) printf( "Problems with rotTXmat \n");

/* Storing all the element mass matrices */

		    for( j = 0; j < neqlsq; ++j )
		    {
			*(mass + neqlsq*k + j) = *(M_el + j);
		    }
		}
	}

	if(lumped_mass_flag)
	{
/* Contract the global mass matrix using the id array only if lumped
   mass is used. */

	    counter = 0;
	    for( i = 0; i < dof ; ++i )
	    {
		/*printf("%5d  %16.8e\n", i, *(mass+i));*/
		if( *(id + i ) > -1 )
		{
		    *(mass + counter ) = *(mass + i );
		    ++counter;
		}
	    }
	}

	return 1;
}


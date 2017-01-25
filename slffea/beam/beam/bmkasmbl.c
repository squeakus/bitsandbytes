/*
    This utility function assembles the stiffness matrix for a finite 
    element program which does analysis on a beam.

	      Updated 11/6/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

    To change the number of integration points from 3 to 2, do:

       :90,600s/(x3+/(x+/g
       :90,600s/(w3+/(w+/g   
       :90,600s/num_int3/num_int/g

    You also have to modify(for 1 point integration):

       wXjacob = 2.0*jacob/2.0;

    To change the number of integration points from 2 to 3, do:

       :90,600s/(x+/(x3+/g
       :90,600s/(w+/(w3+/g
       :90,600s/num_int/num_int3/g
        
    You also have to modify(for 1 point integration):

       wXjacob = 2.0*jacob/3.0;

    

 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bmconst.h"
#include "bmstruct.h"
#include "bmshape_struct.h"

#define SMALL      1.e-20

#define INTEGRATED_K     0
#define FULL_K           0
#define HUGHES_K         0

extern int analysis_flag, dof, numel, numnp, neqn, sof;
extern int gauss_stress_flag;
extern int LU_decomp_flag, numel_K, numel_P, numel_surf;
extern double x[num_int], w[num_int];
extern double x1[num_int1], w1[num_int1], x3[num_int3], w3[num_int3],
	x4[num_int4], w4[num_int4], x_node[num_int];

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int rotTXmat(double *, double *, double *, int, int);

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int bmshape(SHAPE *, double , double , double );

int bmnormcrossX(double *, double *, double *);

int bmKassemble(double *A, BOUND bc, int *connect, double *coord, CURVATURE *curve,
	double *dist_load, int *el_matl, int *el_type, double *force, int *id, int *idiag,
	double *K_diag, int *lm, double *length, double *local_xyz, MATL *matl,
	MOMENT *moment, STRAIN *strain, STRESS *stress, double *U)
{
	int i, i1, i2, j, k, ij, dof_el[neqel];
	int check, counter, dum, dist_flag, node, node0, node1;
	int matl_num, el_num, type_num;
	double area, areaSy, areaSz, Emod, EmodXarea, wXjacob, EmodXIy, EmodXIz,
		G, GXIp, GXareaSy, GXareaSz, phiy, phiz, shear_coeffy, shear_coeffz,
		shear_coeffy2, shear_coeffz2;
	double L, Lsq, Lcube, axis_x[nsd], axis_y[nsd];
	double B[soBRM], DB[soBRM], jacob, fdum, fdum2;
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq], rotate[nsdsq];
	double force_el_dist[neqel], force_gl_dist[neqel], force_el_U[neqel],
		force_local[neqel], U_el[neqel], U_local[neqel],
		vecdum[neqel], vecdum2[neqel];
	double stress_el[sdimRM], strain_el[sdimRM], stress_0[2], strain_0[2];
	SHAPE sh, sh_node;

	dum = 0;
	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		type_num = *(el_type+k);
		Emod = matl[matl_num].E;
		area = matl[matl_num].area;
		areaSy = matl[matl_num].areaSy;
		areaSz = matl[matl_num].areaSz;
		EmodXarea = matl[matl_num].E*matl[matl_num].area;
		EmodXIy = matl[matl_num].E*matl[matl_num].Iy;
		EmodXIz = matl[matl_num].E*matl[matl_num].Iz;
		G = matl[matl_num].E/(1.0 + matl[matl_num].nu)/2.0;
		GXIp = G*(matl[matl_num].Iy + matl[matl_num].Iz);
		GXareaSy = G*matl[matl_num].areaSy;
		GXareaSz = G*matl[matl_num].areaSz;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		L = *(length + k);
		Lsq = L*L;
		Lcube = L*Lsq;

/* The phiy and phyz below are not angles.  They are used to be consistent with the
   notation in Przemieniecki on page 79 in "Theory of Matrix Structural Analysis".
*/
		phiy = 0.0;
		phiz = 0.0;
		if( areaSy > 0.0e-10 && areaSz > 0.0e-10 )
		{
		    phiy = 12.0*EmodXIz/(GXareaSy*Lsq);
		    phiz = 12.0*EmodXIy/(GXareaSz*Lsq);
		}
		else if( type_num == 3) type_num = 2;

		if( type_num == 2 || type_num == 1) 
		{
		    areaSy = area;
		    areaSz = area;
		    phiy = 0.0;
		    phiz = 0.0;
		}

		if( areaSy < SMALL ) areaSy = area;
		if( areaSz < SMALL ) areaSz = area;

/* For the shear_coeffy and shear_coeffz variables below which will be used in
   the calculation of the hermitian beam [DB] matrix, I have a -1.0/(phiy + 1.0)
   term which I have to include to match the results of Przemieniecki.  In
   Przemieniecki, the derivaition of [K] is made by inspection rather than
   variational calculus.  To make the variational derivation, I tried to use
   equation 13.34 in Chapter 13 of a book I found on the Internet. 
   The address is:

      http://caswww.colorado.edu/courses.d/IFEM.d/IFEM.Ch13.d/IFEM.Ch13.pdf

   Felippa, Carlos A., Non-Linear Finite Element Methods, University of Colorado,
	Boulder, 2001.
   carlos.felippa@colorado.edu

   I could not make the derivation work without including the -1.0/(phi + 1.0).
   See my notes file for further discussion.  This book was being worked on when
   I got the chapter so changes to the book may have been made since then.

   Note that the 2 factors below, shear_coeffy and shear_coeffz, were from,
   based on the reference above:

        G*As*gamma^2 = G*As*(phi*v'''*L^2/12.0)^2
                     = G*As*phi^2*v'''^2*L^4/144.0
                     = G*As*(12.0*Emod*I/(G*As*L^2))*phi*v'''^2*L^4/144.0
                     = Emod*I*phi*v'''^2*L^2/12.0

   where:

  gamma = shear stress
     As = effective shear area
   v''' = d^3 v/dx^3 = third derivative with respect to x of displacement

  Again, note that the above needs a factor of -1.0/(phi + 1.0).
   
*/

		shear_coeffy = -EmodXIz*phiy*Lsq/12.0/(phiy + 1.0);
		shear_coeffz = -EmodXIy*phiz*Lsq/12.0/(phiz + 1.0);
		/*shear_coeffy = 0.0;
		shear_coeffz = 0.0;*/

		GXareaSy = G*areaSy;
		GXareaSz = G*areaSz;

/* For the shear_coeffy2 and shear_coeffz2 variables, see page 378 of Hughes.
   This substitution of G*areaS with EmodXIz*12.0/Lsq/(phi + 1.0) will produce
   results exactly the same as the C1 element.  These are for the C0 element.
*/
		/*shear_coeffy2 = GXareaSy;
		shear_coeffz2 = GXareaSz;*/
		shear_coeffy2 = EmodXIz*12.0/Lsq/(phiy + 1.0);
		shear_coeffz2 = EmodXIy*12.0/Lsq/(phiz + 1.0);

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

/* defining the components of an el element vector */

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

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(force_el_dist,0,neqel*sof);
		memset(force_gl_dist,0,neqel*sof);
		memset(force_el_U,0,neqel*sof);
		memset(K_temp,0,neqlsq*sof);

/* assemble the distributed load */

		if(analysis_flag == 1)
		{
		    if( k == bc.dist_load[dum] )
		    {
/* The loop below calculates the points of the gaussian integration
   for the distributed loads */

			for( j = 0; j < num_int; ++j )
			{
			    check = bmshape(&sh, *(x+j), L, Lsq);
			    if(!check) printf( "Problems with bmshape \n");

			    wXjacob = *(w+j)*jacob;

			    el_num = bc.dist_load[dum];
			    *(force_el_dist) += 0.0;
			    *(force_el_dist+1) +=
				sh.N[0].dx0*(*(dist_load+2*el_num))*wXjacob;
			    *(force_el_dist+2) +=
				sh.N[0].dx0*(*(dist_load+2*el_num+1))*wXjacob;
			    *(force_el_dist+3) += 0.0;
			    *(force_el_dist+4) -=
				sh.N[1].dx0*(*(dist_load+2*el_num+1))*wXjacob;
			    *(force_el_dist+5) +=
				sh.N[1].dx0*(*(dist_load+2*el_num))*wXjacob;
			    *(force_el_dist+6) += 0.0;
			    *(force_el_dist+7) +=
				sh.N[2].dx0*(*(dist_load+2*el_num))*wXjacob;
			    *(force_el_dist+8) +=
				sh.N[2].dx0*(*(dist_load+2*el_num+1))*wXjacob;
			    *(force_el_dist+9) += 0.0;
			    *(force_el_dist+10) -=
				sh.N[3].dx0*(*(dist_load+2*el_num+1))*wXjacob;
			    *(force_el_dist+11) +=
				sh.N[3].dx0*(*(dist_load+2*el_num))*wXjacob;
			    /*printf ("bc. dist_load %d %d %d %f \n", i, el_num, bc.dist_load[dum],
				*(dist_load+2*dum));*/
			}
		    }

		}

/* Assembly of the local stiffness matrix:

   For a truss, the only non-zero components are those for the axial DOF terms.  So
   leave as zero in [K_el] everything except *(K_el) and *(K_el+6)
   and *(K_el+72) and *(K_el+78) if the type_num = 1.

   Below, I use the stiffness matrix as given by "Theory of Matrix Structural
   Analysis" by J. S. Przemieniecki on page 79.  The [K] given is for a
   Reissner-Mindlin(type_num == 3) shear deformable beam, although it becomes an
   Euler-Bernoulli(type_num == 2) beam if phiy and phiz are zero.  Because the stiffness
   is derived by inspection, there is no integration involved.
   
*/
		if(type_num < 5)
		{
/* row 0 */
		    *(K_el) = EmodXarea/L;
		    *(K_el+6) = - EmodXarea/L;
		    *(K_el+72) = *(K_el+6);
/* row 6 */
		    *(K_el+78) = EmodXarea/L;
		}

		if(type_num == 2)
		{
/* row 1 */
			*(K_el+13) = 12.0*EmodXIz/Lcube;

			*(K_el+17) = 6.0*EmodXIz/Lsq;
			*(K_el+61) = *(K_el+17);

			*(K_el+19) = -12.0*EmodXIz/Lcube;
			*(K_el+85) = *(K_el+19);

			*(K_el+23) = 6.0*EmodXIz/Lsq;
			*(K_el+133) = *(K_el+23);
/* row 2 */
			*(K_el+26) = 12.0*EmodXIy/Lcube;

			*(K_el+28) = -6.0*EmodXIy/Lsq;
			*(K_el+50) = *(K_el+28);

			*(K_el+32) = -12.0*EmodXIy/Lcube;
			*(K_el+98) = *(K_el+32);

			*(K_el+34) = -6.0*EmodXIy/Lsq;
			*(K_el+122) = *(K_el+34);
/* row 3 */
			*(K_el+39) = GXIp/L;

			*(K_el+45) = -GXIp/L;
			*(K_el+111) = *(K_el+45);
/* row 4 */
			*(K_el+52) = 4.0*EmodXIy/L;

			*(K_el+56) = 6.0*EmodXIy/Lsq;
			*(K_el+100) = *(K_el+56);

			*(K_el+58) = 2.0*EmodXIy/L;
			*(K_el+124) = *(K_el+58);
/* row 5 */
			*(K_el+65) = 4.0*EmodXIz/L;

			*(K_el+67) = -6.0*EmodXIz/Lsq;
			*(K_el+89) = *(K_el+67);

			*(K_el+71) = 2.0*EmodXIz/L;
			*(K_el+137) = *(K_el+71);
/* row 7 */
			*(K_el+91) = 12.0*EmodXIz/Lcube;

			*(K_el+95) = -6.0*EmodXIz/Lsq;
			*(K_el+139) = *(K_el+95);
/* row 8 */
			*(K_el+104) = 12.0*EmodXIy/Lcube; 

			*(K_el+106) = 6.0*EmodXIy/Lsq;
			*(K_el+128) = *(K_el+106);
/* row 9 */
			*(K_el+117) = GXIp/L;
/* row 10 */
			*(K_el+130) = 4.0*EmodXIy/L;
/* row 11 */
			*(K_el+143) = 4.0*EmodXIz/L;
		}

/* The code below works for both element 2 and 3, because it is the same as
   the above when phiy = 0.0 and phiz = 0.0.  So I could have the line:

		if(type_num == 2 || type_num == 3 )

   and get rid of the code above.  The reason I don't do it is because I don't
   want to repeatedly add 0.0.
*/

		if( type_num == 3 )
		{
/* row 1 */
			*(K_el+13) = 12.0*EmodXIz/(1.0 + phiy)/Lcube;

			*(K_el+17) = 6.0*EmodXIz/(1.0 + phiy)/Lsq;
			*(K_el+61) = *(K_el+17);

			*(K_el+19) = -12.0*EmodXIz/(1.0 + phiy)/Lcube;
			*(K_el+85) = *(K_el+19);

			*(K_el+23) = 6.0*EmodXIz/(1.0 + phiy)/Lsq;
			*(K_el+133) = *(K_el+23);
/* row 2 */
			*(K_el+26) = 12.0*EmodXIy/(1.0 + phiz)/Lcube;

			*(K_el+28) = -6.0*EmodXIy/(1.0 + phiz)/Lsq;
			*(K_el+50) = *(K_el+28);

			*(K_el+32) = -12.0*EmodXIy/(1.0 + phiz)/Lcube;
			*(K_el+98) = *(K_el+32);

			*(K_el+34) = -6.0*EmodXIy/(1.0 + phiz)/Lsq;
			*(K_el+122) = *(K_el+34);
/* row 3 */
			*(K_el+39) = GXIp/L;

			*(K_el+45) = -GXIp/L;
			*(K_el+111) = *(K_el+45);
/* row 4 */
			*(K_el+52) = (4.0 + phiz)*EmodXIy/(1.0 + phiz)/L;

			*(K_el+56) = 6.0*EmodXIy/(1.0 + phiz)/Lsq;
			*(K_el+100) = *(K_el+56);

			*(K_el+58) = (2.0 - phiz)*EmodXIy/(1.0 + phiz)/L;
			*(K_el+124) = *(K_el+58);
/* row 5 */
			*(K_el+65) = (4.0 + phiy)*EmodXIz/(1.0 + phiy)/L;

			*(K_el+67) = -6.0*EmodXIz/(1.0 + phiy)/Lsq;
			*(K_el+89) = *(K_el+67);

			*(K_el+71) = (2.0 - phiy)*EmodXIz/(1.0 + phiy)/L;
			*(K_el+137) = *(K_el+71);
/* row 7 */
			*(K_el+91) = 12.0*EmodXIz/(1.0 + phiy)/Lcube;

			*(K_el+95) = -6.0*EmodXIz/(1.0 + phiy)/Lsq;
			*(K_el+139) = *(K_el+95);
/* row 8 */
			*(K_el+104) = 12.0*EmodXIy/(1.0 + phiz)/Lcube; 

			*(K_el+106) = 6.0*EmodXIy/(1.0 + phiz)/Lsq;
			*(K_el+128) = *(K_el+106);
/* row 9 */
			*(K_el+117) = GXIp/L;
/* row 10 */
			*(K_el+130) = (4.0 + phiz)*EmodXIy/(1.0 + phiz)/L;
/* row 11 */
			*(K_el+143) = (4.0 + phiy)*EmodXIz/(1.0 + phiy)/L;
		}

		if(type_num == 4)
		{
/* The code below is for a hinged element as defined by equation (13.31) on page
   13-14 in chapter 13 of the reference "Non-Linear Finite Element Methods" by
   Carlos A. Felippa.
*/

/* row 1 */
			*(K_el+13) = 12.0*EmodXIz/Lcube;

			*(K_el+17) = 6.0*EmodXIz/Lsq;
			*(K_el+61) = *(K_el+17);

			*(K_el+19) = -12.0*EmodXIz/Lcube;
			*(K_el+85) = *(K_el+19);

			*(K_el+23) = 6.0*EmodXIz/Lsq;
			*(K_el+133) = *(K_el+23);
/* row 2 */
			*(K_el+26) = 12.0*EmodXIy/Lcube;

			*(K_el+28) = -6.0*EmodXIy/Lsq;
			*(K_el+50) = *(K_el+28);

			*(K_el+32) = -12.0*EmodXIy/Lcube;
			*(K_el+98) = *(K_el+32);

			*(K_el+34) = -6.0*EmodXIy/Lsq;
			*(K_el+122) = *(K_el+34);
/* row 4 */
			*(K_el+52) = 3.0*EmodXIy/L;

			*(K_el+56) = 6.0*EmodXIy/Lsq;
			*(K_el+100) = *(K_el+56);

			*(K_el+58) = 3.0*EmodXIy/L;
			*(K_el+124) = *(K_el+58);
/* row 5 */
			*(K_el+65) = 3.0*EmodXIz/L;

			*(K_el+67) = -6.0*EmodXIz/Lsq;
			*(K_el+89) = *(K_el+67);

			*(K_el+71) = 3.0*EmodXIz/L;
			*(K_el+137) = *(K_el+71);
/* row 7 */
			*(K_el+91) = 12.0*EmodXIz/Lcube;

			*(K_el+95) = -6.0*EmodXIz/Lsq;
			*(K_el+139) = *(K_el+95);
/* row 8 */
			*(K_el+104) = 12.0*EmodXIy/Lcube; 

			*(K_el+106) = 6.0*EmodXIy/Lsq;
			*(K_el+128) = *(K_el+106);
/* row 10 */
			*(K_el+130) = 3.0*EmodXIy/L;
/* row 11 */
			*(K_el+143) = 3.0*EmodXIz/L;
		}


/* The loop below calculates the points of 1 point gaussian integration
   for the stiffness matrices which are based on linear(C0) interpolating
   functions.  These beam element stiffnesses are integrated
   using 1 point Gauss.  For an explanation of the differences between
   C0 and C1 interpolating functions, see page 109 and 110 of Tom Hughes
   in "The Finite Element Method".
*/


		if(type_num == 5 || type_num == 6)
		{
		    for( j = 0; j < num_int1; ++j )
		    {
			memset(B,0,soBRM*sof);
			memset(DB,0,soBRM*sof);
			memset(K_local,0,neqlsq*sof);

			check = bmshape(&sh, *(x1+j), L, Lsq);
			if(!check) printf( "Problems with bmshape \n");

/* Assembly of the local stiffness matrix:

   For a truss, the only non-zero components are those for
   the axial loads.  So leave as zero in [B] and [DB] everything except
   *(B) and *(B+6) and *(DB) and *(DB+6) if the type_num = 1.
*/

			*(B) = sh.Nhat[0].dx1;
			*(B+6) = sh.Nhat[1].dx1;
			*(DB) = EmodXarea*sh.Nhat[0].dx1;
			*(DB+6) = EmodXarea*sh.Nhat[1].dx1;
			if(type_num == 6)
			{
/* The B matrix */
			    *(B+17) = sh.Nhat[0].dx1;
			    *(B+23) = sh.Nhat[1].dx1;
			    *(B+28) = sh.Nhat[0].dx1;
			    *(B+34) = sh.Nhat[1].dx1;
			    *(B+39) = sh.Nhat[0].dx1;
			    *(B+45) = sh.Nhat[1].dx1;
			    *(B+50) = sh.Nhat[0].dx1;
			    *(B+52) = sh.Nhat[0].dx0;
			    *(B+56) = sh.Nhat[1].dx1;
			    *(B+58) = sh.Nhat[1].dx0;
			    *(B+61) = sh.Nhat[0].dx1;
			    *(B+65) = -sh.Nhat[0].dx0;
			    *(B+67) = sh.Nhat[1].dx1;
			    *(B+71) = -sh.Nhat[1].dx0;

/* The DB matrix */
			    *(DB+17) = EmodXIz*sh.Nhat[0].dx1;
			    *(DB+23) = EmodXIz*sh.Nhat[1].dx1;
			    *(DB+28) = EmodXIy*sh.Nhat[0].dx1;
			    *(DB+34) = EmodXIy*sh.Nhat[1].dx1;
			    *(DB+39) = GXIp*sh.Nhat[0].dx1;
			    *(DB+45) = GXIp*sh.Nhat[1].dx1;
			    *(DB+50) = shear_coeffz2*sh.Nhat[0].dx1;
			    *(DB+52) = shear_coeffz2*sh.Nhat[0].dx0;
			    *(DB+56) = shear_coeffz2*sh.Nhat[1].dx1;
			    *(DB+58) = shear_coeffz2*sh.Nhat[1].dx0;
			    *(DB+61) = shear_coeffy2*sh.Nhat[0].dx1;
			    *(DB+65) = -shear_coeffy2*sh.Nhat[0].dx0;
			    *(DB+67) = shear_coeffy2*sh.Nhat[1].dx1;
			    *(DB+71) = -shear_coeffy2*sh.Nhat[1].dx0;
			}

			check = matXT(K_local, B, DB, neqel, neqel, sdimRM);
			if(!check) printf( "Problems with matXT \n");

			wXjacob = *(w1+j)*jacob;

			for( i1 = 0; i1 < neqlsq; ++i1 )
			{
			    *(K_el + i1) += *(K_local + i1)*wXjacob;
			}
		    }
		}

/* The loop below calculates the points of 2 point gaussian integration
   for the stiffness matrices which are based on Hermitian(C1) interpolating
   functions.  These beam element stiffnesses are integrated using 2
   point Gauss.
*/

		if(type_num > 6)
		{
		    for( j = 0; j < num_int; ++j )
		    {
			memset(B,0,soBRM*sof);
			memset(DB,0,soBRM*sof);
			memset(K_local,0,neqlsq*sof);

			check = bmshape(&sh, *(x+j), L, Lsq);
			if(!check) printf( "Problems with bmshape \n");

/* Assembly of the local stiffness matrix:

   For a truss, the only non-zero components are those for
   the axial loads.  So leave as zero in [B] and [DB] everything except
   *(B) and *(B+6) and *(DB) and *(DB+6) if the type_num = 1.
*/

			*(B) = sh.Nhat[0].dx1;
			*(B+6) = sh.Nhat[1].dx1;
			*(DB) = EmodXarea*sh.Nhat[0].dx1;
			*(DB+6) = EmodXarea*sh.Nhat[1].dx1;
			if(type_num > 7)
			{
/* The B matrix */
			    *(B+13) = sh.N[0].dx2;
			    *(B+17) = sh.N[1].dx2;
			    *(B+19) = sh.N[2].dx2;
			    *(B+23) = sh.N[3].dx2;
			    *(B+26) = -sh.N[0].dx2;
			    *(B+28) = sh.N[1].dx2;
			    *(B+32) = -sh.N[2].dx2;
			    *(B+34) = sh.N[3].dx2;
			    *(B+39) = sh.Nhat[0].dx1;
			    *(B+45) = sh.Nhat[1].dx1;

			    *(B+50) = -sh.N[0].dx3;
			    *(B+52) = sh.N[1].dx3;
			    *(B+56) = -sh.N[2].dx3;
			    *(B+58) = sh.N[3].dx3;
			    *(B+61) = sh.N[0].dx3;
			    *(B+65) = sh.N[1].dx3;
			    *(B+67) = sh.N[2].dx3;
			    *(B+71) = sh.N[3].dx3;

/* The DB matrix */
			    *(DB+13) = EmodXIz*sh.N[0].dx2;
			    *(DB+17) = EmodXIz*sh.N[1].dx2;
			    *(DB+19) = EmodXIz*sh.N[2].dx2;
			    *(DB+23) = EmodXIz*sh.N[3].dx2;
			    *(DB+26) = -EmodXIy*sh.N[0].dx2;
			    *(DB+28) = EmodXIy*sh.N[1].dx2;
			    *(DB+32) = -EmodXIy*sh.N[2].dx2;
			    *(DB+34) = EmodXIy*sh.N[3].dx2;
			    *(DB+39) = GXIp*sh.Nhat[0].dx1;
			    *(DB+45) = GXIp*sh.Nhat[1].dx1;

			    *(DB+50) = -shear_coeffz*sh.N[0].dx3;
			    *(DB+52) = shear_coeffz*sh.N[1].dx3;
			    *(DB+56) = -shear_coeffz*sh.N[2].dx3;
			    *(DB+58) = shear_coeffz*sh.N[3].dx3;
			    *(DB+61) = shear_coeffy*sh.N[0].dx3;
			    *(DB+65) = shear_coeffy*sh.N[1].dx3;
			    *(DB+67) = shear_coeffy*sh.N[2].dx3;
			    *(DB+71) = shear_coeffy*sh.N[3].dx3;
			}

			check = matXT(K_local, B, DB, neqel, neqel, sdimRM);
			if(!check) printf( "Problems with matXT \n");

			wXjacob = *(w+j)*jacob;

			for( i1 = 0; i1 < neqlsq; ++i1 )
			{
			    *(K_el + i1) += *(K_local + i1)*wXjacob;
			}
		    }
		}


		dist_flag = 0;
		if( k == bc.dist_load[dum] )
		{
			++dum;
			dist_flag = 1;
		}

/* Put K back to global coordinates */

		check = matXrot(K_temp, K_el, rotate, neqel, neqel);
		if(!check) printf( "Problems with matXrot \n");

		check = rotTXmat(K_el, rotate, K_temp, neqel, neqel);
		if(!check) printf( "Problems with rotTXmat \n");

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(force_el_U, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

		  for( j = 0; j < neqel; ++j )
		  {
			*(force + *(dof_el+j)) -= *(force_el_U + j);
		  }

/* Compute the equivalant nodal forces based on distributed element loads */

		  if( dist_flag )
		  {
		     check = rotTXmat(force_gl_dist, rotate, force_el_dist, 1, neqel);
		     for( j = 0; j < neqel; ++j )
		     {
			*(force + *(dof_el+j)) += *(force_gl_dist + j);
		     }
		  }
/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

		  if(LU_decomp_flag)
		  {
			check = globalKassemble(A, idiag, K_el, (lm + k*neqel),
				neqel);
			if(!check) printf( "Problems with globalKassemble \n");
		  }
		  else
		  {
			check = globalConjKassemble(A, dof_el, k, K_diag, K_el,
				neqel, neqlsq, numel_K);
			if(!check) printf( "Problems with globalConjKassemble \n");
		  }
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel; ++j )
			{
				*(force + *(dof_el+j)) += *(force_el_U + j);
			}

/* Put force back into local coordinates */

			check = rotXmat(force_local, rotate, force_el_U, 1, neqel);
			if(!check) printf( "Problems with rotXmat \n");

			/*for( j = 0; j < neqel; ++j )
			{
				printf("\n dof %3d %3d %14.6f ", k, j, *(force_local + j));
			}
			fdum = *(force_local)/area;
			printf("\n %3d %3d %14.6f ", k, j, fdum);*/

			if(type_num < 5)
			{
/* Update of the global stress matrix */

			    stress[k].pt[0].xx += -*(force_local)/area;
			    stress[k].pt[0].xy += -*(force_local+1)/areaSy;
			    stress[k].pt[0].zx += -*(force_local+2)/areaSz;
			    moment[k].pt[0].xx += -*(force_local+3);
			    moment[k].pt[0].yy += -*(force_local+4);
			    moment[k].pt[0].zz += -*(force_local+5);
			    stress[k].pt[1].xx += *(force_local+6)/area;
			    stress[k].pt[1].xy += *(force_local+7)/areaSy;
			    stress[k].pt[1].zx += *(force_local+8)/areaSz;
			    moment[k].pt[1].xx += *(force_local+9);
			    moment[k].pt[1].yy += *(force_local+10);
			    moment[k].pt[1].zz += *(force_local+11);

/* Update of the global strain matrix */

			    strain[k].pt[0].xx = stress[k].pt[0].xx/Emod;
			    strain[k].pt[0].xy = stress[k].pt[0].xy/G;
			    strain[k].pt[0].zx = stress[k].pt[0].zx/G;
			    curve[k].pt[0].zz = moment[k].pt[0].zz/EmodXIz;
			    curve[k].pt[0].yy = moment[k].pt[0].yy/EmodXIy;
			    curve[k].pt[0].xx = moment[k].pt[0].xx/GXIp;
			    strain[k].pt[1].xx = stress[k].pt[1].xx/Emod;
			    strain[k].pt[1].xy = stress[k].pt[1].xy/G;
			    strain[k].pt[1].zx = stress[k].pt[1].zx/G;
			    curve[k].pt[1].zz = moment[k].pt[1].zz/EmodXIz;
			    curve[k].pt[1].yy = moment[k].pt[1].yy/EmodXIy;
			    curve[k].pt[1].xx = moment[k].pt[1].xx/GXIp;
			}

/* Calculate the element local U matrix */

			check = rotXmat(U_local, rotate, U_el, 1, neqel);
			if(!check) printf( "Problems with rotXmat \n");

/* The loop below calculates either the points of 1 point gaussian integration
   or at the nodes for the [B] and [stress] matrices which are based on linear(C0)
   interpolating functions. 
*/

			if(type_num == 5 || type_num == 6)
			{
			    memset(vecdum,0,neqel*sof);
			    for( j = 0; j < num_int; ++j )
			    {
				memset(B,0,soBRM*sof);
				memset(stress_el,0,sdimRM*sof);
				memset(strain_el,0,sdimRM*sof);
				memset(vecdum2,0,neqel*sof);

				gauss_stress_flag = 1;

/* Calculate sh at integration point */
				check = bmshape(&sh, *(x1), L, Lsq);
				if(!check) printf( "Problems with bmshape \n");

/* Assembly of the local stiffness matrix */

/* Similarly for what was discussed above:
   For a truss, the only non-zero components are those for
   the axial loads.  So leave as zero in [B] everything except
   *(B) and *(B+6) if the type_num = 1.
*/
				if(gauss_stress_flag)
				{
				    *(B) = sh.Nhat[0].dx1;
				    *(B+6) = sh.Nhat[1].dx1;
				    if(type_num == 6)
				    {
					*(B+17) = sh.Nhat[0].dx1;
					*(B+23) = sh.Nhat[1].dx1;
					*(B+28) = sh.Nhat[0].dx1;
					*(B+34) = sh.Nhat[1].dx1;
					*(B+39) = sh.Nhat[0].dx1;
					*(B+45) = sh.Nhat[1].dx1;
					*(B+50) = sh.Nhat[0].dx1;
					*(B+52) = sh.Nhat[0].dx0;
					*(B+56) = sh.Nhat[1].dx1;
					*(B+58) = sh.Nhat[1].dx0;
					*(B+61) = sh.Nhat[0].dx1;
					*(B+65) = -sh.Nhat[0].dx0;
					*(B+67) = sh.Nhat[1].dx1;
					*(B+71) = -sh.Nhat[1].dx0;
				    }
				}
				else
				{
				    *(B) = sh_node.Nhat[0].dx1;
				    *(B+6) = sh_node.Nhat[1].dx1;
				    if(type_num == 6)
				    {
					*(B+17) = sh_node.Nhat[0].dx1;
					*(B+23) = sh_node.Nhat[1].dx1;
					*(B+28) = sh_node.Nhat[0].dx1;
					*(B+34) = sh_node.Nhat[1].dx1;
					*(B+39) = sh_node.Nhat[0].dx1;
					*(B+45) = sh_node.Nhat[1].dx1;
					*(B+50) = sh_node.Nhat[0].dx1;
					*(B+52) = sh_node.Nhat[0].dx0;
					*(B+56) = sh_node.Nhat[1].dx1;
					*(B+58) = sh_node.Nhat[1].dx0;
					*(B+61) = sh_node.Nhat[0].dx1;
					*(B+65) = -sh_node.Nhat[0].dx0;
					*(B+67) = sh_node.Nhat[1].dx1;
					*(B+71) = -sh_node.Nhat[1].dx0;
				    }
				}

/* Calculation of the local strain matrix */

				check=matX(strain_el, B, U_local, sdimRM, 1, neqel );
				if(!check) printf( "Problems with matX\n");

/* Update of the global strain matrix */

				strain[k].pt[j].xx = *(strain_el);
				curve[k].pt[j].zz  = *(strain_el+1);
				curve[k].pt[j].yy  = *(strain_el+2);
				curve[k].pt[j].xx  = *(strain_el+3);
				strain[k].pt[j].zx = *(strain_el+4);
				strain[k].pt[j].xy = *(strain_el+5);

/* Calculation of the local stress matrix */

				*(stress_el) = strain[k].pt[j].xx*Emod;
				*(stress_el+1) = curve[k].pt[j].zz*EmodXIz;
				*(stress_el+2) = curve[k].pt[j].yy*EmodXIy;
				*(stress_el+3) = curve[k].pt[j].xx*GXIp;
				*(stress_el+4) = strain[k].pt[j].zx*shear_coeffz2;
				*(stress_el+5) = strain[k].pt[j].xy*shear_coeffy2;

/* Update of the global stress matrix */

				fdum = 1.0;
				if(j == 1) fdum = -1.0;

				stress[k].pt[j].xx += *(stress_el);
				moment[k].pt[j].zz += *(stress_el+1) + *(stress_el+5)*fdum*L/2.0;
				moment[k].pt[j].yy += *(stress_el+2) - *(stress_el+4)*fdum*L/2.0;
				moment[k].pt[j].xx += *(stress_el+3);
				stress[k].pt[j].zx += *(stress_el+4)/areaSz;
				stress[k].pt[j].xy += *(stress_el+5)/areaSy;


			    }

			}


/* The loop below calculates either the points of 2 point gaussian integration
   or at the nodes for the [B] and [stress] matrices which are based on Hermitian(C1)
   interpolating functions. 
*/

			if(type_num > 6)
			{

/* Calculation of the local strain matrix at the beam center */

			    *(strain_0) = -*(U_local+5)/L + *(U_local+11)/L;
			    *(strain_0+1) = -*(U_local+4)/L + *(U_local+10)/L;

			    *(stress_0) = *(strain_0)*EmodXIz;
			    *(stress_0+1) = *(strain_0+1)*EmodXIy;

			    for( j = 0; j < num_int; ++j )
			    {
				memset(B,0,soBRM*sof);
				memset(stress_el,0,sdimRM*sof);
				memset(strain_el,0,sdimRM*sof);

				if(gauss_stress_flag)
				{
/* Calculate sh at integration point */
				    check = bmshape(&sh, *(x+j), L, Lsq);
				    if(!check) printf( "Problems with bmshape \n");
				}
				else
				{
/* Calculate sh at nodal point */
				    check = bmshape(&sh_node, *(x_node+j), L, Lsq);
				    if(!check) printf( "Problems with bmshape \n");
				}

/* Similarly for what was discussed above:
   For a truss, the only non-zero components are those for
   the axial loads.  So leave as zero in [B] everything except
   *(B) and *(B+6) if the type_num = 1.
*/
				if(gauss_stress_flag)
				{
				    *(B) = sh.Nhat[0].dx1;
				    *(B+6) = sh.Nhat[1].dx1;
				    if(type_num > 7)
				    {
					*(B+13) = sh.N[0].dx2;
					*(B+17) = sh.N[1].dx2;
					*(B+19) = sh.N[2].dx2;
					*(B+23) = sh.N[3].dx2;
					*(B+26) = -sh.N[0].dx2;
					*(B+28) = sh.N[1].dx2;
					*(B+32) = -sh.N[2].dx2;
					*(B+34) = sh.N[3].dx2;
					*(B+39) = sh.Nhat[0].dx1;
					*(B+45) = sh.Nhat[1].dx1;
					*(B+50) = -sh.N[0].dx3;
					*(B+52) = sh.N[1].dx3;
					*(B+56) = -sh.N[2].dx3;
					*(B+58) = sh.N[3].dx3;
					*(B+61) = sh.N[0].dx3;
					*(B+65) = sh.N[1].dx3;
					*(B+67) = sh.N[2].dx3;
					*(B+71) = sh.N[3].dx3;
				    }
				}
				else
				{
				    *(B) = sh_node.Nhat[0].dx1;
				    *(B+6) = sh_node.Nhat[1].dx1;
				    if(type_num > 7)
				    {
					*(B+13) = sh_node.N[0].dx2;
					*(B+17) = sh_node.N[1].dx2;
					*(B+19) = sh_node.N[2].dx2;
					*(B+23) = sh_node.N[3].dx2;
					*(B+26) = -sh_node.N[0].dx2;
					*(B+28) = sh_node.N[1].dx2;
					*(B+32) = -sh_node.N[2].dx2;
					*(B+34) = sh_node.N[3].dx2;
					*(B+39) = sh_node.Nhat[0].dx1;
					*(B+45) = sh_node.Nhat[1].dx1;
					*(B+50) = -sh_node.N[0].dx3;
					*(B+52) = sh_node.N[1].dx3;
					*(B+56) = -sh_node.N[2].dx3;
					*(B+58) = sh_node.N[3].dx3;
					*(B+61) = sh_node.N[0].dx3;
					*(B+65) = sh_node.N[1].dx3;
					*(B+67) = sh_node.N[2].dx3;
					*(B+71) = sh_node.N[3].dx3;
				    }
				}

/* Calculation of the local strain matrix */

				check=matX(strain_el, B, U_local, sdimRM, 1, neqel );
				if(!check) printf( "Problems with matX\n");

/* Update of the global strain matrix */

				strain[k].pt[j].xx = *(strain_el);
				curve[k].pt[j].zz  = *(strain_el+1);
				curve[k].pt[j].yy  = *(strain_el+2);
				curve[k].pt[j].xx  = *(strain_el+3);
				strain[k].pt[j].zx = *(strain_el+4)*phiz*Lsq/12.0;
				strain[k].pt[j].xy = *(strain_el+5)*phiy*Lsq/12.0;

/* Calculation of the local stress matrix */
/* Below, the units for *(stress_el+4) and *(stress_el+5) are N*m^2 which represent neither
   force, moment or stress.  the reason for this is explained in the notes file */

				*(stress_el) = strain[k].pt[j].xx*Emod;
				*(stress_el+1) = curve[k].pt[j].zz*EmodXIz;
				*(stress_el+2) = curve[k].pt[j].yy*EmodXIy;
				*(stress_el+3) = curve[k].pt[j].xx*GXIp;
				*(stress_el+4) = -strain[k].pt[j].zx*EmodXIy/(1.0 + phiz);
				*(stress_el+5) = -strain[k].pt[j].xy*EmodXIz/(1.0 + phiy);

/* Update of the global stress matrix */

				fdum = 1.0;
				if(j == 1) fdum = -1.0;

				stress[k].pt[j].xx += *(stress_el);
				moment[k].pt[j].zz += *(stress_el+1) - *(stress_el+5)*fdum*6.0/L;
				moment[k].pt[j].yy += *(stress_el+2) - *(stress_el+4)*fdum*6.0/L;
				moment[k].pt[j].xx += *(stress_el+3);
				stress[k].pt[j].zx += *(stress_el+4)*12.0/Lsq/areaSz;
				stress[k].pt[j].xy += -*(stress_el+5)*12.0/Lsq/areaSy;

			    }
			}

		}


	}

	if(analysis_flag == 1)
	{

/* Contract the global force matrix using the id array only if LU decomposition
   is used. */

	  if(LU_decomp_flag)
	  {
	     counter = 0;
	     for( i = 0; i < dof ; ++i )
	     {
		if( *(id + i ) > -1 )
		{
			*(force + counter ) = *(force + i );
			++counter;
		}
	     }
	  }
	}

	return 1;
}


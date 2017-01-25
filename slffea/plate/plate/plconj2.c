/*
    This utility function uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite element
    program which does analysis on a triangle plate.  The first
    assembles the P matrix.  It is called by the second function
    which allocates the memory.  These go with the calculation of
    displacement.

		Updated 11/4/06

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
#include "plconst.h"
#include "plstruct.h"

#define SMALL      1.e-20

extern int analysis_flag, dof, numel, numnp, sof, plane_stress_flag, flag_3D,
	flag_quad_element;
extern int LU_decomp_flag, numel_K, numel_P;
extern SH shg, shl;
extern double w[num_int+1], *Area0;
extern int  iteration_max, iteration_const, iteration;
extern double tolerance;

int rotTXmat(double *, double *, double *, int, int);
    
int rotXmat2(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int); 

int plateBtr( double *, double *);

int plateB1pt( double *, double *);
    
int triangleB( double *,double *);
    
int trshg( double *, int, int, double *, double *, double *);

int dotX(double *, double *, double *, int);

int plBoundary( double *, BOUND );


int plConjPassemble_triangle(double *A, int *connect, double *coord, int *el_matl,
	double *local_xyz, MATL *matl, double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 11/1/06
*/
	int i, i1, i2, j, k, dof_el6[neqel18], sdof_el[npel3*nsd];
	int check, node, dum;
	int matl_num;
	double Emod, Pois, G1, G2, G3, thickness, shearK, const1, const2;
	double fdum, fdum2;
	double D11,D12,D21,D22,D11mem,D12mem,D21mem,D22mem,Gmem;
	double lamda, mu;
	double B[soBtr], DB[soBtr], Bmem[soBmemtr], DBmem[soBmemtr];
	double K_temp[neqlsq324], K_el[neqlsq324], K_local[neqlsq81];
	double K_bend[neqlsq81], K_mem[neqlsq36];
	double rotate[nsdsq];
	double U_el[neqel18];
	double coord_el[npel3*nsd], coord_el_trans[npel3*nsd],
		coord_el_local[npel3*nsd2], coord_el_local_trans[npel3*nsd2];
	double X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3, X12, X13, Y12, Y13, Z12, Z13;
	double det[1], wXdet, area_el;
	double P_el[neqel18];


	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel_K; ++k )
	{

		for( j = 0; j < npel3; ++j )
		{
			node = *(connect+npel3*k+j);

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;
		}


/* Assembly of the global P matrix */

		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, (A+k*neqlsq324), U_el, neqel18, 1, neqel18);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel18; ++j )
		{
			*(P_global_CG+*(dof_el6+j)) += *(P_el+j);
		}
	}

	for( k = numel_K; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;
		shearK = matl[matl_num].shear;

/* Due to the problem of shear locking, I temporarily divided the shear by 320.0
   but have now removed this.

		shearK = matl[matl_num].shear/320.0;
*/

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* Recalculate lamda for plane stress */

		if(plane_stress_flag)
			lamda = Emod*Pois/(1.0-Pois*Pois);

/* FOR THE MEMBRANE */

		D11mem = thickness*(lamda+2.0*mu);
		D12mem = thickness*lamda;
		D21mem = thickness*lamda;
		D22mem = thickness*(lamda+2.0*mu);

		Gmem = thickness*mu;

/* FOR THE BENDING */

		const1 = thickness*thickness*thickness/12.0;
		const2 = thickness*shearK;

		/*printf("lamda, mu, Emod, Pois %f %f %f %f \n", lamda, mu, Emod, Pois);*/

		D11 = (lamda+2.0*mu)*const1;
		D12 = lamda*const1;
		D21 = lamda*const1;
		D22 = (lamda+2.0*mu)*const1;
		G1 = mu*const1;
		G2 = mu*const2;
		G3 = mu*const2;

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Create the coord_el vector for one element */

		for( j = 0; j < npel3; ++j )
		{

			node = *(connect+npel3*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel3*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel3*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;

		}

		memset(U_el,0,neqel18*sof);
		memset(K_el,0,neqlsq81*sof);
		memset(K_temp,0,neqlsq324*sof);
		memset(K_bend,0,neqlsq81*sof);
		memset(K_mem,0,neqlsq36*sof);
		memset(rotate,0,nsdsq*sof);
		
		if(!flag_3D)
		{
/* For 2-D meshes */
		    X1 = *(coord_el_trans);
		    X2 = *(coord_el_trans + 1);
		    X3 = *(coord_el_trans + 2);

		    Y1 = *(coord_el_trans + npel3*1);
		    Y2 = *(coord_el_trans + npel3*1 + 1);
		    Y3 = *(coord_el_trans + npel3*1 + 2);
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
		    *(rotate + 6) = *(local_xyz + nsdsq*k + 6);
		    *(rotate + 7) = *(local_xyz + nsdsq*k + 7);
		    *(rotate + 8) = *(local_xyz + nsdsq*k + 8);

/* Put coord_el into local coordinates */

		    dum = nsd*npel3;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");

/* Assembly of the B matrix.  This is taken from "Fundamentals of the Finite
   Element Method" by Hartley Grandin Jr., page 201-205.

     It should be noted that I have permutated the node sequences.
     (see Hughes Figure 3.I.5 on page 167; node 3 in Hughes is node 1
     in SLFFEA, node 2 is node 3, and node 1 goes to node 2.)
     This is because I want to be consistant with the tetrahedron.  You can
     read more about this change in teshl for the tetrahedron.

     This change only effects the calculation of the area.  The [B] matrix
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

		memset(B,0,soBtr*sof);
		memset(DB,0,soBtr*sof);
		memset(Bmem,0,soBmemtr*sof);
		memset(DBmem,0,soBmemtr*sof);

		memset(K_local,0,neqlsq81*sof);


/* FOR THE MEMBRANE */

/* For [Bmem] below, see "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204.  Despite the permutation of nodes mentioned above,
   B remains the same.  */

		*(Bmem) = Y2 - Y3;
		*(Bmem+2) = Y3 - Y1;
		*(Bmem+4) = Y1 - Y2;
		*(Bmem+7) = X3 - X2;
		*(Bmem+9) = X1 - X3;
		*(Bmem+11) = X2 - X1;
		*(Bmem+12) = X3 - X2;
		*(Bmem+13) = Y2 - Y3;
		*(Bmem+14) = X1 - X3;
		*(Bmem+15) = Y3 - Y1;
		*(Bmem+16) = X2 - X1;
		*(Bmem+17) = Y1 - Y2;


/* FOR THE BENDING */

/* See "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204 for a description of how the differences in
   Xs and Ys translate into shape function derivatives.   As you can see, these
   derivatives are constant.

   Also, look on page 13-3 to 13-6 in the "ANSYS User's Manual".  It
   discusses integration points for triangles.  For 1 point integration,
   r = s = t = 1/3 and the weight is 1.
*/

		fdum = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);
		area_el = .5*fdum;

		*(B+2) = (Y3 - Y2)/fdum;
		*(B+5) = (Y1 - Y3)/fdum;
		*(B+8) = (Y2 - Y1)/fdum;
		*(B+10) = (X3 - X2)/fdum;
		*(B+13) = (X1 - X3)/fdum;
		*(B+16) = (X2 - X1)/fdum;
		*(B+19) = (Y2 - Y3)/fdum;
		*(B+20) = (X2 - X3)/fdum;
		*(B+22) = (Y3 - Y1)/fdum;
		*(B+23) = (X3 - X1)/fdum;
		*(B+25) = (Y1 - Y2)/fdum;
		*(B+26) = (X1 - X2)/fdum;

		*(B+27) = (Y2 - Y3)/fdum;
		*(B+29) = 1.0/3.0;
		*(B+30) = (Y3 - Y1)/fdum;
		*(B+32) = 1.0/3.0;
		*(B+33) = (Y1 - Y2)/fdum;
		*(B+35) = 1.0/3.0;

		*(B+36) = (X3 - X2)/fdum;
		*(B+37) = -1.0/3.0;
		*(B+39) = (X1 - X3)/fdum;
		*(B+40) = -1.0/3.0;
		*(B+42) = (X2 - X1)/fdum;
		*(B+43) = -1.0/3.0;

/* FOR THE MEMBRANE */

		for( i1 = 0; i1 < neqel6; ++i1 )
		{
			*(DBmem+i1) = *(Bmem+i1)*D11mem+
				*(Bmem+neqel6*1+i1)*D12mem;
			*(DBmem+neqel6*1+i1) = *(Bmem+i1)*D21mem+
				*(Bmem+neqel6*1+i1)*D22mem;
			*(DBmem+neqel6*2+i1) = *(Bmem+neqel6*2+i1)*Gmem;
		}

		check=matXT(K_local, Bmem, DBmem, neqel6, neqel6, sdim3);
		if(!check) printf( "Problems with matXT  \n");

		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_mem + j) = *(K_local + j)/( 4.0*area_el );
		}

/* FOR THE BENDING */

		for( i1 = 0; i1 < neqel9; ++i1 )
		{
			*(DB+i1) = *(B+i1)*D11 +
				*(B+neqel9*1+i1)*D12;
			*(DB+neqel9*1+i1) = *(B+i1)*D21 +
				*(B+neqel9*1+i1)*D22;
			*(DB+neqel9*2+i1) = *(B+neqel9*2+i1)*G1;
			*(DB+neqel9*3+i1) = *(B+neqel9*3+i1)*G2;
			*(DB+neqel9*4+i1) = *(B+neqel9*4+i1)*G3;
		}

		check=matXT(K_local, B, DB, neqel9, neqel9, sdim5);
		if(!check) printf( "Problems with matXT \n");

		for( j = 0; j < neqlsq81; ++j )
		{
			*(K_bend + j) = *(K_local + j)*area_el;
		}

/* Unlike the quadrilateral and triangle membrane elements, the lines below are executed
   for both 2-D and 3-D elements because I wanted to make the rotation easier to understand.
   This element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out K_bend to test if I can get the same results
   as the triangle element which has no bending.  Also, I need to zero out K_bend in
   "plmasmbl.c" with:

                memset(M_bend,0,neqlsq81*sof);
*/
#if 0
		memset(K_bend,0,neqlsq81*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel3; ++i )
		{
			for( j = 0; j < npel3; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof6*neqel18*i + ndof6*j) =
				*(K_mem + ndof2*neqel6*i + ndof2*j);
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 1) =
				*(K_mem + ndof2*neqel6*i + ndof2*j + 1);
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j) =
				*(K_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j);
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 1) =
				*(K_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j + 1);
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 5) = 0.0;
			}
/* The line below is an attempt to add an artificial stiffness to the rotation in local z.
   The triangle plate has much more problems than the quad plate for modal analysis, so I
   added this to improve results.
*/ 
			*(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*i + 5) = 10.0e20;
		}
		memcpy(K_el, K_temp, neqlsq324*sizeof(double));

		if(flag_3D)
		{
/* For 3-D meshes */

/* Put K back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

		   check = matXrot(K_temp, K_el, rotate, neqel18, neqel18);
		   if(!check) printf( "Problems with matXrot \n");

		   check = rotTXmat(K_el, rotate, K_temp, neqel18, neqel18);
		   if(!check) printf( "Problems with rotTXmat \n");
		}

		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, K_el, U_el, neqel18, 1, neqel18);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel18; ++j )
		{
			*(P_global_CG+*(dof_el6+j)) += *(P_el+j);
		}
	}

	return 1;
}



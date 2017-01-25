/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a plate.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

		Updated 11/1/06

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

int matXrot(double *, double *, double *, int, int);

int rotTXmat(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int matX(double *,double *, double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int plateB4pt( double *, double *);

int plateB4pt_node( double *, double *);

int plshg( double *, int, SH, SH, double *);

int dotX(double *, double *, double *, int);

int plBoundary( double *, BOUND );

int plConjPassemble_triangle(double *, int *, double *, int *, double *, MATL *,
	double *, double *);


int plConjPassemble(double *A, int *connect, double *coord, int *el_matl,
	double *local_xyz, MATL *matl, double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 11/1/06
*/
	int i, i1, i2, j, k, dof_el6[neqel24], sdof_el[npel4*nsd];
	int check, node, dum;
	int matl_num;
	double Emod, Pois, G1, G2, G3, thickness, shearK, const1, const2;
	double fdum, fdum2;
	double D11,D12,D21,D22,D11mem,D12mem,D21mem,D22mem,Gmem;
	double lamda, mu;
	double B[soB], DB[soB], Bmem[soBmem], DBmem[soBmem];
	double K_temp[neqlsq576], K_el[neqlsq576], K_local[neqlsq144];
	double K_bend[neqlsq144], K_mem[neqlsq64];
	double rotate[nsdsq];
	double U_el[neqel24];
	double coord_el[npel4*nsd], coord_el_trans[npel4*nsd],
		coord_el_local[npel4*nsd2], coord_el_local_trans[npel4*nsd2];
	double det[num_int+1], wXdet;
	double P_el[neqel24];


	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel_K; ++k )
	{

		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;
		}


/* Assembly of the global P matrix */

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, (A+k*neqlsq576), U_el, neqel24, 1, neqel24);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel24; ++j )
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

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* The lamda below is for plane stress */

		if(plane_stress_flag) lamda = Emod*Pois/(1.0-Pois*Pois);

		/*printf("lamda, mu, Emod, Pois %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* FOR THE MEMBRANE */

		D11mem = thickness*(lamda+2.0*mu);
		D12mem = thickness*lamda;
		D21mem = thickness*lamda;
		D22mem = thickness*(lamda+2.0*mu);

		Gmem = thickness*mu;

/* FOR THE BENDING */

		const1 = thickness*thickness*thickness/12.0;
		const2 = thickness*shearK;

		D11 = (lamda+2.0*mu)*const1;
		D12 = lamda*const1;
		D21 = lamda*const1;
		D22 = (lamda+2.0*mu)*const1;
		G1 = mu*const1;
		G2 = mu*const2;
		G3 = mu*const2;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel4*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel4*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;
		}

		memset(rotate,0,nsdsq*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel4 + j) = *(coord_el_trans + npel4*1 + j);
		    }
		}
		else
		{
/* For 3-D quad meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  The local basis used for the
   rotation has already been calculated and stored in local_xyz[].  Below, it is
   copied to rotate[].

   The rotation matrix, rotate[], is used to rotate the global U_el, made up of
   nodal displacements x, y, and z and rotation angels x, y, and z into the local x, y,
   and z coordinates of the plane of the plate.  It should be noted that the rotation
   angle z doesn't really exist, but is added to make the rotation multiplications easier.
   I can now take advantage of the beam and truss rotation functions by doing this.
   This rotation is also used for the rotations of the element stiffnesses.  
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

/* Rotate the global coordinates into the local x and y coordinates of the plane of the plate.
*/
		    dum = nsd*npel4;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel4*1 + j) = *(coord_el_local+nsd2*j+1);
		    }
		}

/* Assembly of the shg matrix for each integration point */

		check=plshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with plshg \n");

		memset(U_el,0,neqel24*sof);
		memset(K_el,0,neqlsq576*sof);
		memset(K_temp,0,neqlsq576*sof);
		memset(K_bend,0,neqlsq144*sof);
		memset(K_mem,0,neqlsq64*sof);

/* Calculate the lower part of the B and DB matrices for 1X1 point gaussian integration */

		memset((B+neqel12*(sdim5-2)),0,neqel12*(sdim5-3)*sof);
		memset((DB+neqel12*(sdim5-2)),0,neqel12*(sdim5-3)*sof);

		check = plateB1pt(shg.shear, (B+(sdim5-2)*neqel12));
		if(!check) printf( "Problems with plateB1pt \n");
		for( i1 = 0; i1 < neqel12; ++i1 )
		{
			*(DB+neqel12*3+i1) = *(B+neqel12*3+i1)*G2;
			*(DB+neqel12*4+i1) = *(B+neqel12*4+i1)*G3;
		}

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		for( j = 0; j < num_int; ++j )
		{

/* FOR THE MEMBRANE */

		    memset(Bmem,0,soBmem*sof);
		    memset(DBmem,0,soBmem*sof);
		    memset(K_local,0,neqlsq64*sof);

/* Assembly of the Bmem matrix */

		    check = quadB((shg.bend+npel4*(nsd2+1)*j), Bmem);
		    if(!check) printf( "Problems with quadB \n");

		    for( i1 = 0; i1 < neqel8; ++i1 )
		    {
			*(DBmem+i1) = *(Bmem+i1)*D11mem+
				*(Bmem+neqel8*1+i1)*D12mem;
			*(DBmem+neqel8*1+i1) = *(Bmem+i1)*D21mem+
				*(Bmem+neqel8*1+i1)*D22mem;
			*(DBmem+neqel8*2+i1) = *(Bmem+neqel8*2+i1)*Gmem;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, Bmem, DBmem, neqel8, neqel8, sdim3);
		    if(!check) printf( "Problems with matXT  \n");
		    for( i2 = 0; i2 < neqlsq64; ++i2 )
		    {
			  *(K_mem+i2) += *(K_local+i2)*wXdet;
		    }


/* FOR THE BENDING */

		    memset(B,0,neqel12*(sdim5-2)*sof);
		    memset(DB,0,neqel12*(sdim5-2)*sof);
		    memset(K_local,0,neqlsq144*sof);

/* Calculate the upper part of the B and DB matrices for 4X4 point gaussian integration */

		    check = plateB4pt((shg.bend+npel4*(nsd2+1)*j), B);
		    if(!check) printf( "Problems with plateB4pt \n");

		    for( i1 = 0; i1 < neqel12; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11 +
				*(B+neqel12*1+i1)*D12;
			*(DB+neqel12*1+i1) = *(B+i1)*D21 +
				*(B+neqel12*1+i1)*D22;
			*(DB+neqel12*2+i1) = *(B+neqel12*2+i1)*G1;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, B, DB, neqel12, neqel12, sdim5);
		    if(!check) printf( "Problems with matXT \n");

/* Note that I'm using the 4X4 determinant and weight for the 1X1 Gauss integration
   which is added 4 times.  This is a valid operation which can be proven. */

		    for( i2 = 0; i2 < neqlsq144; ++i2 )
		    {
			  *(K_bend+i2) += *(K_local+i2)*wXdet;
		    }
		}

/*
		for( i1 = 0; i1 < neqel8; ++i1 )
		{
		   for( i2 = 0; i2 < neqel8; ++i2 )
		   {
			printf("%12.5e ",*(K_mem+neqel8*i1+i2));
		   }
		   printf("\n ");
		}
		printf("\n ");
*/

/*
		for( i1 = 0; i1 < neqel12; ++i1 )
		{
		   for( i2 = 0; i2 < neqel12; ++i2 )
		   {
			printf("%12.5e ",*(K_bend+neqel12*i1+i2));
		   }
		   printf("\n ");
		}
		printf("\n ");
*/

/* Unlike the quadrilateral and triangle membrane elements, the lines below are executed
   for both 2-D and 3-D elements because I wanted to make the rotation easier to understand.
   This element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out K_bend to test if I can get the same results
   as the quadrilateral element which has no bending.   Also, I need to zero out K_bend in
   "plmasmbl.c" with:

                memset(M_bend,0,neqlsq144*sof);

   Note that if an eigenmode comparison is made, I must change the num_eigen in "fempl.c" to:

      num_eigen = (int)(2.0*nmode);

   to match the quad.
*/
#if 0
		memset(K_bend,0,neqlsq144*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel4; ++i )
		{
			for( j = 0; j < npel4; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof6*neqel24*i + ndof6*j) =
				*(K_mem + ndof2*neqel8*i + ndof2*j);
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 1) =
				*(K_mem + ndof2*neqel8*i + ndof2*j + 1);
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j) =
				*(K_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j);
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 1) =
				*(K_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j + 1);
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 5) = 0.0;
			}
		}
		memcpy(K_el, K_temp, neqlsq576*sizeof(double));

		if(flag_3D)
		{
/* For 3-D meshes */

/* Put K back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

		   check = matXrot(K_temp, K_el, rotate, neqel24, neqel24);
		   if(!check) printf( "Problems with matXrot \n");

		   check = rotTXmat(K_el, rotate, K_temp, neqel24, neqel24);
		   if(!check) printf( "Problems with rotTXmat \n");
		}

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6 + j));
		}


		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, K_el, U_el, neqel24, 1, neqel24);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel24; ++j )
		{
			*(P_global_CG+*(dof_el6+j)) += *(P_el+j);
		}
	}

	return 1;
}


int plConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, double *local_xyz, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to plConjPassemble to get the
   product of [A]*[p].

                        Updated 11/1/06

   It is taken from the algorithm 10.3.1 given in "Matrix Computations",
   by Golub, page 534.
*/
	int i, j, sofmf, ptr_inc;
	int check, counter;
	double *mem_double;
	double *p, *P_global_CG, *r, *z;
	double alpha, alpha2, beta;
	double fdum, fdum2;

/* For the doubles */
	sofmf = 4*dof;
	mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
		printf( "failed to allocate memory for doubles\n ");
		exit(1);
	}

/* For the Conjugate Gradient Method doubles */

	                                        ptr_inc = 0;
	p=(mem_double+ptr_inc);                 ptr_inc += dof;
	P_global_CG=(mem_double+ptr_inc);       ptr_inc += dof;
	r=(mem_double+ptr_inc);                 ptr_inc += dof;
	z=(mem_double+ptr_inc);                 ptr_inc += dof;

/* Using Conjugate gradient method to find displacements */

	memset(P_global_CG,0,dof*sof);
	memset(p,0,dof*sof);
	memset(r,0,dof*sof);
	memset(z,0,dof*sof);

	for( j = 0; j < dof; ++j )
	{
		*(K_diag + j) += SMALL;
		*(r+j) = *(force+j);
		*(z + j) = *(r + j)/(*(K_diag + j));
		*(p+j) = *(z+j);
	}
	check = plBoundary (r, bc);
	if(!check) printf( " Problems with plBoundary \n");

	check = plBoundary (p, bc);
	if(!check) printf( " Problems with plBoundary \n");

	alpha = 0.0;
	alpha2 = 0.0;
	beta = 0.0;
	fdum2 = 1000.0;
	counter = 0;
	check = dotX(&fdum, r, z, dof);

	printf("\n iteration %3d iteration max %3d \n", iteration, iteration_max);
	/*for( iteration = 0; iteration < iteration_max; ++iteration )*/
	while(fdum2 > tolerance && counter < iteration_max )
	{

		printf( "\n %3d %16.8e\n",counter, fdum2);
		if( !flag_quad_element )
		{
/* Triangle elements */
		    check = plConjPassemble_triangle( A, connect, coord, el_matl,
			local_xyz, matl, P_global_CG, p);
		}
		else
		{
/* Quad elements */
		    check = plConjPassemble( A, connect, coord, el_matl, local_xyz,
			matl, P_global_CG, p);
		}
		if(!check) printf( " Problems with plConjPassemble \n");
		check = plBoundary (P_global_CG, bc);
		if(!check) printf( " Problems with plBoundary \n");
		check = dotX(&alpha2, p, P_global_CG, dof);	
		alpha = fdum/(SMALL + alpha2);

		for( j = 0; j < dof; ++j )
		{
		    /*printf( "%4d %14.5e  %14.5e  %14.5e  %14.5e  %14.5e %14.5e\n",j,alpha,
			beta,*(U+j),*(r+j),*(P_global_CG+j),*(p+j));*/
		    *(U+j) += alpha*(*(p+j));
		    *(r+j) -=  alpha*(*(P_global_CG+j));
		    *(z + j) = *(r + j)/(*(K_diag + j));
		}

		check = dotX(&fdum2, r, z, dof);
		beta = fdum2/(SMALL + fdum);
		fdum = fdum2;
		
		for( j = 0; j < dof; ++j )
		{
		    /*printf("\n  %3d %12.7f  %14.5f ",j,*(U+j),*(P_global_CG+j));*/
		    /*printf( "%4d %14.5f  %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
		    printf( "%4d %14.8f  %14.8f  %14.8f  %14.8f %14.8f\n",j,
			*(U+j)*beta,*(r+j)*beta,*(P_global_CG+j)*alpha,
			*(force+j)*alpha);*/
		    *(p+j) = *(z+j)+beta*(*(p+j));
		}
		check = plBoundary (p, bc);
		if(!check) printf( " Problems with plBoundary \n");

		++counter;
	}

	if(counter > iteration_max - 1 )
	{
		printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",counter,fdum2);
		printf( "Problem may not have converged during Conj. Grad.\n");
	}
/*
The lines below are for testing the quality of the calculation:

1) r should be 0.0
2) P_global_CG( = A*U ) - force should be 0.0
*/

#if 0
	if( !flag_quad_element )
	{
/* Triangle elements */
	    check = plConjPassemble_triangle( A, connect, coord, el_matl, local_xyz,
		matl, P_global_CG, U);
	    if(!check) printf( " Problems with plConjPassemble \n");
	}
	else
	{
/* Quad elements */
	    check = plConjPassemble( A, connect, coord, el_matl, local_xyz, matl,
		P_global_CG, U);
	    if(!check) printf( " Problems with plConjPassemble \n");
	}

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
#endif

	free(mem_double);

	return 1;
}



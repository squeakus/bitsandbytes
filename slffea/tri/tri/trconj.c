/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a triangle.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

	        Updated 11/25/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

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

#define SMALL      1.e-20

extern int analysis_flag, dof, numel, numnp, plane_stress_flag, sof, flag_3D;
extern int LU_decomp_flag, numel_K, numel_P;
extern double shg[sosh], shl[sosh], w[num_int], *Area0;
extern int  iteration_max, iteration_const, iteration;
extern double tolerance;

int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int triangleB(double *,double *);

int dotX(double *, double *, double *, int);

int Boundary( double *, BOUND );


int trConjPassemble(double *A, int *connect, double *coord, int *el_matl,
	double *local_xyz, MATL *matl, double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 11/25/08
*/
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness;
	double fdum, area_el;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq36], rotate[nsd2*nsd];
	double U_el[neqel];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double X1, X2, X3, Y1, Y2, Y3;
	double det[num_int];
	double P_el[neqel];


	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel_K; ++k )
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

		check = matX(P_el, (A+k*neqlsq), U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	for( k = numel_K; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* Recalculate lamda for plane stress */

		if(plane_stress_flag)
			lamda = Emod*Pois/(1.0-Pois*Pois);

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Normally, with plane elements, we assume a unit thickness in the transverse direction.  But
   because these elements can be 3 dimensional, I multiply the material property matrix by the
   thickness.  This is justified by equation 5.141a in "Theory of Matrix Structural
   Analysis" by J. S. Przemieniecki on page 86.
*/

		D11 = thickness*(lamda+2.0*mu);
		D12 = thickness*lamda;
		D21 = thickness*lamda;
		D22 = thickness*(lamda+2.0*mu);

		G = mu;
		Gt = thickness*mu;

/* Create the coord_el vector for one element */

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

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		area_el = .5*fdum;

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(K_temp,0,neqlsq*sof);

		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);
		memset(K_local,0,neqlsq36*sof);
	
		*(B) = Y2 - Y3;
		*(B+2) = Y3 - Y1;
		*(B+4) = Y1 - Y2;
		*(B+7) = X3 - X2;
		*(B+9) = X1 - X3;
		*(B+11) = X2 - X1;
		*(B+12) = X3 - X2;
		*(B+13) = Y2 - Y3;
		*(B+14) = X1 - X3;
		*(B+15) = Y3 - Y1;
		*(B+16) = X2 - X1;
		*(B+17) = Y1 - Y2;

		for( i1 = 0; i1 < neqel6; ++i1 )
		{
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel6*1+i1)*D12;
			*(DB+neqel6*1+i1) = *(B+i1)*D21+
				*(B+neqel6*1+i1)*D22;
			*(DB+neqel6*2+i1) = *(B+neqel6*2+i1)*Gt;
		}

		check=matXT(K_local, B, DB, neqel6, neqel6, sdim);
		if(!check) printf( "Problems with matXT  \n");

		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_el + j) = *(K_local + j)/(4.0*area_el);
		}

		if(!flag_3D)
		{
/* For 2-D meshes */
		   for( i = 0; i < npel; ++i )
		   {
			for( j = 0; j < npel; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof*neqel*i + ndof*j) =
				*(K_el + ndof2*neqel6*i + ndof2*j);
			   *(K_temp + ndof*neqel*i + ndof*j + 1) =
				*(K_el + ndof2*neqel6*i + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof*neqel*i + neqel  + ndof*j) =
				*(K_el + ndof2*neqel6*i + neqel6 + ndof2*j);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(K_el + ndof2*neqel6*i + neqel6 + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 2) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j) = 0.0;
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 1) = 0.0;
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 2) = 0.0;
			}
		   }
		   memcpy(K_el, K_temp, neqlsq*sizeof(double));
		}
		else
		{
/* For 3-D meshes */

/* Put K back to global coordinates */

		   check = matXrot2(K_temp, K_el, rotate, neqel6, neqel);
		   if(!check) printf( "Problems with matXrot2 \n");

		   check = rotTXmat2(K_el, rotate, K_temp, neqel, neqel);
		   if(!check) printf( "Problems with rotTXmat2 \n");
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	return 1;
}


int trConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, double *local_xyz, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to trConjPassemble to get the
   product of [A]*[p].

                        Updated 1/24/06

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
	check = Boundary (r, bc);
	if(!check) printf( " Problems with Boundary \n");

	check = Boundary (p, bc);
	if(!check) printf( " Problems with Boundary \n");

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
		check = trConjPassemble( A, connect, coord, el_matl, local_xyz, matl,
			P_global_CG, p);
		if(!check) printf( " Problems with trConjPassemble \n");
		check = Boundary (P_global_CG, bc);
		if(!check) printf( " Problems with Boundary \n");
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
		check = Boundary (p, bc);
		if(!check) printf( " Problems with Boundary \n");

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

/*
	check = trConjPassemble( A, connect, coord, el_matl, local_xyz, matl,
		P_global_CG, U);
	if(!check) printf( " Problems with trConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}



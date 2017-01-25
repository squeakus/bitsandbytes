/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a truss.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

                 Updated 11/8/06

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
#include "tsconst.h"
#include "tsstruct.h"

#define SMALL      1.e-20

extern int analysis_flag, dof, numel, numnp, sof;
extern int LU_decomp_flag, numel_K, numel_P;
extern int iteration_max, iteration_const, iteration;
extern double tolerance;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int dotX(double *, double *, double *, int);

int Boundary( double *, BOUND );


int tsConjPassemble(double *A, int *connect, double *coord, int *el_matl, double *length,
	double *local_xyz, MATL *matl, double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 11/7/06
*/
	int i, i1, i2, j, k, dof_el[neqel];
	int check, node, node0, node1;
	int matl_num;
	double Emod, area, EmodXarea, wXjacob;
	double fdum, fdum2;
	double L, Lsq;
	double B[soB], DB[soB], jacob;
	double K_temp[npel*neqel], K_el[neqlsq], K_local[npelsq], rotate[npel*neqel];
	double U_el[neqel];
	double x[num_int], w[num_int];
	double P_el[neqel];


	memset(P_global_CG,0,dof*sof);

	*(x)=0.0;
	*(w)=2.0;

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
		area = matl[matl_num].area;
		EmodXarea = Emod*area;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		L = *(length + k);
		Lsq = L*L;

		jacob = L/2.0;

/* Assembly of the rotation matrix.  This is taken from equation 5.42 on page 69 of:

     Przemieniecki, J. S., Theory of Matrix Structural Analysis, Dover
        Publications Inc., New York, 1985.
*/

		memset(rotate,0,npel*neqel*sof);

		*(rotate)    = *(local_xyz + nsdsq*k);
		*(rotate+1)  = *(local_xyz + nsdsq*k + 1);
		*(rotate+2)  = *(local_xyz + nsdsq*k + 2);
		*(rotate+9)  = *(local_xyz + nsdsq*k);
		*(rotate+10) = *(local_xyz + nsdsq*k + 1);
		*(rotate+11) = *(local_xyz + nsdsq*k + 2);

/* defining the components of an element vector */

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(K_temp,0,npel*neqel*sof);
		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);
		memset(K_local,0,npel*npel*sof);

/* Assembly of the local stiffness matrix */

		*(B) = - 1.0/L;
		*(B+1) = 1.0/L;

		*(DB) = Emod*(*(B));
		*(DB+1) = Emod*(*(B+1));

		*(K_local) = EmodXarea/L/L;
		*(K_local+1) = - EmodXarea/L/L;
		*(K_local+2) = - EmodXarea/L/L;
		*(K_local+3) = EmodXarea/L/L;

		wXjacob = *(w)*jacob;
		for( i1 = 0; i1 < npelsq; ++i1 )
		{
		    *(K_el + i1) += *(K_local + i1)*wXjacob;
		}

/* Put K back to global coordinates */

		check = matX(K_temp, K_el, rotate, npel, neqel, npel);
		if(!check) printf( "Problems with matX \n");

		check = matXT(K_el, rotate, K_temp, neqel, neqel, npel);
		if(!check) printf( "Problems with matXT \n");

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


int tsConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, double *length, double *local_xyz,
	MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to tsConjPassemble to get the
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
		check = tsConjPassemble( A, connect, coord, el_matl, length, local_xyz,
			matl, P_global_CG, p);
		if(!check) printf( " Problems with tsConjPassemble \n");
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
	check = tsConjPassemble( A, connect, coord, el_matl, length, local_xyz, matl,
		P_global_CG, U);
	if(!check) printf( " Problems with tsConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}



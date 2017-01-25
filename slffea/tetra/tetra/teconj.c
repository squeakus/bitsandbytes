/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a tetrahedron.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

	        Updated 11/2/06

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
#include "teconst.h"
#include "testruct.h"

#define SMALL      1.e-20

extern int analysis_flag, dof, numel, numnp, sof;
extern int LU_decomp_flag, numel_K, numel_P;
extern double shg[sosh], shl[sosh], w[num_int], *Vol0;
extern int  iteration_max, iteration_const, iteration;
extern double tolerance;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int tetraB(double *,double *);

int dotX(double *, double *, double *, int);

int Boundary( double *, BOUND );


int teConjPassemble(double *A, int *connect, double *coord, int *el_matl, MATL *matl,
	double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 12/18/02
*/
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node;
	int matl_num;
	double Emod, Pois, G;
	double fdum, volume_el;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double U_el[neqel];
	double coord_el_trans[npel*nsd];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
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

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));
		mu = Emod/(1.0+Pois)/2.0;

		D11 = lamda+2.0*mu;
		D12 = lamda;
		D13 = lamda;
		D21 = lamda;
		D22 = lamda+2.0*mu;
		D23 = lamda;
		D31 = lamda;
		D32 = lamda;
		D33 = lamda+2.0*mu;

		G = mu;

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* Assembly of the B matrix.

   This is taken from "Fundamentals of the Finite Element Method" by
   Hartley Grandin Jr., page 383-387. 

   I have changed the node numbering to be consistant with the
   numbering of the brick element, specifically the counter-clockwise
   numbering scheme.  This means that node 4 in Grandin is node 1 in
   SLFFEA, node 1 becomes node 2, node 2 is node 3, and node 3 goes
   to node 4.  So X4 -> X1, X1 -> X2, X2 -> X3, X3 -> X4, and similarly
   for Y and Z.  This also means a4 -> a1, a1 -> a2, a2 -> a3, a3 -> a4
   and similarly for b and c.  Everything else is the same.
*/

		X1 = *(coord_el_trans);
		X2 = *(coord_el_trans + 1);
		X3 = *(coord_el_trans + 2);
		X4 = *(coord_el_trans + 3);

		Y1 = *(coord_el_trans + npel*1);
		Y2 = *(coord_el_trans + npel*1 + 1);
		Y3 = *(coord_el_trans + npel*1 + 2);
		Y4 = *(coord_el_trans + npel*1 + 3);

		Z1 = *(coord_el_trans + npel*2);
		Z2 = *(coord_el_trans + npel*2 + 1);
		Z3 = *(coord_el_trans + npel*2 + 2);
		Z4 = *(coord_el_trans + npel*2 + 3);

		a2 = (Y3 - Y1)*(Z4 - Z1) - (Y4 - Y1)*(Z3 - Z1);
		a3 = (Y4 - Y1)*(Z2 - Z1) - (Y2 - Y1)*(Z4 - Z1);
		a4 = (Y2 - Y1)*(Z3 - Z1) - (Y3 - Y1)*(Z2 - Z1);
		a1 = -(a2 + a3 + a4);

		b2 = (X4 - X1)*(Z3 - Z1) - (X3 - X1)*(Z4 - Z1);
		b3 = (X2 - X1)*(Z4 - Z1) - (X4 - X1)*(Z2 - Z1);
		b4 = (X3 - X1)*(Z2 - Z1) - (X2 - X1)*(Z3 - Z1);
		b1 = -(b2 + b3 + b4);

		c2 = (X3 - X1)*(Y4 - Y1) - (X4 - X1)*(Y3 - Y1);
		c3 = (X4 - X1)*(Y2 - Y1) - (X2 - X1)*(Y4 - Y1);
		c4 = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);
		c1 = -(c2 + c3 + c4);

		fdum = (X2 - X1)*a2 + (Y2 - Y1)*b2 + (Z2 - Z1)*c2;

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 6*Vol:%f\n", k, fdum);
			/*return 0;*/
		}

		volume_el = pt1667*fdum;

		memset(K_el,0,neqlsq*sof);
		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);

		*(B) = a1;
		*(B+3) = a2;
		*(B+6) = a3;
		*(B+9) = a4;

		*(B+13) = b1;
		*(B+16) = b2;
		*(B+19) = b3;
		*(B+22) = b4;

		*(B+26) = c1;
		*(B+29) = c2;
		*(B+32) = c3;
		*(B+35) = c4;

		*(B+36) = b1;
		*(B+37) = a1;
		*(B+39) = b2;
		*(B+40) = a2;
		*(B+42) = b3;
		*(B+43) = a3;
		*(B+45) = b4;
		*(B+46) = a4;

		*(B+48) = c1;
		*(B+50) = a1;
		*(B+51) = c2;
		*(B+53) = a2;
		*(B+54) = c3;
		*(B+56) = a3;
		*(B+57) = c4;
		*(B+59) = a4;

		*(B+61) = c1;
		*(B+62) = b1;
		*(B+64) = c2;
		*(B+65) = b2;
		*(B+67) = c3;
		*(B+68) = b3;
		*(B+70) = c4;
		*(B+71) = b4;

		for( i1 = 0; i1 < neqel; ++i1 )
		{
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel*1+i1)*D12+
				*(B+neqel*2+i1)*D13;
			*(DB+neqel*1+i1) = *(B+i1)*D21+
				*(B+neqel*1+i1)*D22+
				*(B+neqel*2+i1)*D23;
			*(DB+neqel*2+i1) = *(B+i1)*D31+
				*(B+neqel*1+i1)*D32+
				*(B+neqel*2+i1)*D33;
			*(DB+neqel*3+i1) = *(B+neqel*3+i1)*G;
			*(DB+neqel*4+i1) = *(B+neqel*4+i1)*G;
			*(DB+neqel*5+i1) = *(B+neqel*5+i1)*G;
		}

		check=matXT(K_el, B, DB, neqel, neqel, sdim);
		if(!check) printf( "Problems with matXT \n");

		for( i2 = 0; i2 < neqlsq; ++i2 )
		{
		    *(K_el+i2) /= 36.0*volume_el;
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


int teConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to teConjPassemble to get the
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
		check = teConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, p);
		if(!check) printf( " Problems with teConjPassemble \n");
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
	check = teConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, U);
	if(!check) printf( " Problems with teConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}



/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a quad.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

		Updated 11/2/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qd3const.h"
#include "qd3struct.h"

#define SMALL      1.e-20

extern int analysis_flag, dof, numed, numel, numnp, plane_stress_flag, sof;
extern int LU_decomp_flag, numel_EM, numel_P;
extern double double dcdx[nsdsq*num_int], shg[sosh], shl[sosh], w[num_int], *Area0;
extern int  iteration_max, iteration_const, iteration;
extern double tolerance;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int quadB(double *,double *);

int qd3shg( double *, double *, int, double *, double *, double *, double *);

int dotX(double *, double *, double *, int);

int qdBoundary( double *, BOUND );


int qd3ConjPassemble(double *A, int *connect, double *coord, int *el_matl, MATL *matl,
	double *P_global, double *U)
{
/* This function assembles the P_global matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

			Updated 7/7/00
*/
        int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node;
	int matl_num;
	double Emod, Pois, G;
	double fdum, fdum2;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double U_el[neqel];
        double coord_el_trans[npel*nsd];
	double det[num_int];
        double P_el[neqel];


	memset(P_global,0,dof*sof);

        for( k = 0; k < numel_EM; ++k )
        {

                for( j = 0; j < npel; ++j )
                {
			node = *(connect+npel*k+j);

                	*(dof_el+ndof*j) = ndof*node;
                	*(dof_el+ndof*j+1) = ndof*node+1;
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
                	*(P_global+*(dof_el+j)) += *(P_el+j);
		}
	}

        for( k = numel_EM; k < numel; ++k )
        {
                matl_num = *(el_matl+k);
                Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* Recalculate lamda for plane stress */

		if(plane_stress_flag)
			lamda = Emod*Pois/(1.0-Pois*Pois);

        	/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

		D11 = lamda+2.0*mu;
		D12 = lamda;
		D21 = lamda;
		D22 = lamda+2.0*mu;

		G = mu;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;

		}


/* Assembly of the shg matrix for each integration point */

		check = qd3shg( dcdx, det, k, shl, shg, coord_el_trans, &fdum);
		if(!check) printf( "Problems with qd3shg \n");

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
                memset(K_el,0,neqlsq*sof);

                for( j = 0; j < num_int; ++j )
                {

                    memset(B,0,soB*sof);
                    memset(DB,0,soB*sof);
                    memset(K_temp,0,neqlsq*sof);

/* Assembly of the B matrix */

		    check = quadB((shg+npel*(nsd+1)*j),B);
                    if(!check) printf( "Problems with quadB \n");

                    for( i1 = 0; i1 < neqel; ++i1 )
                    {
                        *(DB+i1) = *(B+i1)*D11+
				*(B+neqel*1+i1)*D12;
                        *(DB+neqel*1+i1) = *(B+i1)*D21+
				*(B+neqel*1+i1)*D22;
                        *(DB+neqel*2+i1) = *(B+neqel*2+i1)*G;
                    }

                    check = matXT(K_temp, B, DB, neqel, neqel, sdim);
                    if(!check) printf( "Problems with matXT  \n");
                    for( i2 = 0; i2 < neqlsq; ++i2 )
                    {
                          *(K_el+i2) += *(K_temp+i2)*(*(w+j))*(*(det+j));
                    }
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
                	*(P_global+*(dof_el+j)) += *(P_el+j);
		}
        }

        return 1;
}


int qd3ConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *Att_diag, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to qd3ConjPassemble to get the
   product of [A]*[p].

			Updated 1/7/01

   It is taken from the algorithm 10.3.1 given in "Matrix Computations",
   by Golub, page 534.
*/
	int i, j, sofmf, ptr_inc;
	int check, counter;
	double *mem_double;
	double *p, *P_global, *r, *rm1, *z, *zm1;
	double alpha, alpha2, beta;
	double fdum, fdum2;

/* For the doubles */
	sofmf = 6*dof;
	mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
		printf( "failed to allocate memory for doubles\n ");
		exit(1);
	}

/* For the Conjugate Gradient Method doubles */

	                                        ptr_inc = 0;
	p=(mem_double+ptr_inc);                 ptr_inc += dof;
	P_global=(mem_double+ptr_inc);          ptr_inc += dof;
	rm1=(mem_double+ptr_inc);               ptr_inc += dof;
	r=(mem_double+ptr_inc);                 ptr_inc += dof;
	z=(mem_double+ptr_inc);                 ptr_inc += dof;
	zm1=(mem_double+ptr_inc);               ptr_inc += dof;

/* Using Conjugate gradient method to find displacements */

	memset(P_global,0,dof*sof);
	memset(p,0,dof*sof);
	memset(r,0,dof*sof);
	memset(rm1,0,dof*sof);
	memset(z,0,dof*sof);
	memset(zm1,0,dof*sof);

        for( j = 0; j < dof; ++j )
	{
		*(Att_diag + j) += SMALL;
		*(r+j) = *(force+j);
		*(z + j) = *(r + j)/(*(Att_diag + j));
		*(p+j) = *(z+j);
	}
	check = qd3Boundary (r, bc);
	if(!check) printf( " Problems with qd3Boundary \n");

	check = qd3Boundary (p, bc);
	if(!check) printf( " Problems with qd3Boundary \n");

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
		check = qd3ConjPassemble( A, connect, coord, el_matl, matl, P_global, p);
		if(!check) printf( " Problems with qd3ConjPassemble \n");
		check = qd3Boundary (P_global, bc);
		if(!check) printf( " Problems with qd3Boundary \n");
		check = dotX(&alpha2, p, P_global, dof);	
		alpha = fdum/(SMALL + alpha2);

        	for( j = 0; j < dof; ++j )
		{
            	    /*printf( "%4d %14.5e  %14.5e  %14.5e  %14.5e  %14.5e %14.5e\n",j,alpha,
			beta,*(U+j),*(r+j),*(P_global+j),*(p+j));*/
		    *(rm1+j) = *(r + j); 
		    *(zm1+j) = *(z + j); 
    		    *(U+j) += alpha*(*(p+j));
		    *(r+j) -=  alpha*(*(P_global+j));
		    *(z + j) = *(r + j)/(*(Att_diag + j));
		}

		check = dotX(&fdum2, r, z, dof);
		beta = fdum2/(SMALL + fdum);
		fdum = fdum2;
	
        	for( j = 0; j < dof; ++j )
        	{
       		    /*printf("\n  %3d %12.7f  %14.5f ",j,*(U+j),*(P_global+j));*/
            	    /*printf( "%4d %14.5f  %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,
			*(U+j),*(r+j),*(P_global+j),*(force+j));
            	    printf( "%4d %14.8f  %14.8f  %14.8f  %14.8f %14.8f\n",j,
			*(U+j)*beta,*(r+j)*beta,*(P_global+j)*alpha,
			*(force+j)*alpha);*/
            	    *(p+j) = *(z+j)+beta*(*(p+j));

		}
		check = qd3Boundary (p, bc);
		if(!check) printf( " Problems with qd3Boundary \n");

		++counter;
	}

	if(counter > iteration_max - 1 )
	{
		printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",counter,fdum2);
		printf( "Problem may not have converged during Conj. Grad.\n");
	}
/*
	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global+j),*(force+j));
	}
*/

	check = qd3ConjPassemble( A, connect, coord, el_matl, matl, P_global, U);
	if(!check) printf( " Problems with qd3ConjPassemble \n");

	free(mem_double);

	return 1;
}



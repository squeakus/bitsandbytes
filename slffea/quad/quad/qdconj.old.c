/*
    This utility function uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a quad.  The first
    assembles the P matrix.  It is called by the second function
    which allocates the memory.  These go with the calculation of
    displacement.

		Updated 9/8/06

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
#include "qdconst.h"
#include "qdstruct.h"

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

int quadB(double *,double *);

int qdshg( double *, int, double *, double *, double *);

int dotX(double *, double *, double *, int);

int Boundary( double *, BOUND );


int qdConjPassemble(double *A, int *connect, double *coord, int *el_matl, MATL *matl,
	double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 9/1/06
*/
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, dum;
	int matl_num;
	double Emod, Pois, G, fdum1, fdum2, fdum3, fdum4;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq64];
	double rotate[npel*nsd2*nsd];
	double U_el[neqel];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double det[num_int], wXdet;
	double P_el[neqel];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd], vec_dum4[nsd];
	double xp[npel], xp0[npel], xp1[npel], xp2[npel],
		yp[npel], yp0[npel], yp1[npel], yp2[npel],
		zp[npel], zp0[npel], zp1[npel], zp2[npel];

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

		memset(rotate,0,npel*nsd2*nsd*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel + j) = *(coord_el_trans + npel*1 + j);
		    }
		}
		else
		{
/* For 3-D quad meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  To do this I have to calculate
   the normal to the plate face, then cross product that normal with an in plane vector
   to get the other local axis direction.

   I also use the algorithm in my shell code that tries to align this normal vector
   as closely as possible to the global z direction.
*/

/*
   I considered many different methods for calculating the initial local z axis.  For a
   complete discussion of this, read the comments given in "qdkasmbl.c".  This is a legacy
   code for the method I did not use.
*/

		    *(xp2+1) = *(xp1+3) = *(xp0+0) = *(coord_el_trans);
		    *(xp2+2) = *(xp1+0) = *(xp0+1) = *(coord_el_trans + 1);
		    *(xp2+3) = *(xp1+1) = *(xp0+2) = *(coord_el_trans + 2);
		    *(xp2+0) = *(xp1+2) = *(xp0+3) = *(coord_el_trans + 3);

		    *(yp2+1) = *(yp1+3) = *(yp0+0) = *(coord_el_trans + npel*1);
		    *(yp2+2) = *(yp1+0) = *(yp0+1) = *(coord_el_trans + npel*1 + 1);
		    *(yp2+3) = *(yp1+1) = *(yp0+2) = *(coord_el_trans + npel*1 + 2);
		    *(yp2+0) = *(yp1+2) = *(yp0+3) = *(coord_el_trans + npel*1 + 3);

		    *(zp2+1) = *(zp1+3) = *(zp0+0) = *(coord_el_trans + npel*2);
		    *(zp2+2) = *(zp1+0) = *(zp0+1) = *(coord_el_trans + npel*2 + 1);
		    *(zp2+3) = *(zp1+1) = *(zp0+2) = *(coord_el_trans + npel*2 + 2);
		    *(zp2+0) = *(zp1+2) = *(zp0+3) = *(coord_el_trans + npel*2 + 3);

		    for( j = 0; j < npel; ++j )
		    {
			*(vec_dum1)     = *(xp1+j) - *(xp0+j);
			*(vec_dum1 + 1) = *(yp1+j) - *(yp0+j);
			*(vec_dum1 + 2) = *(zp1+j) - *(zp0+j);
			*(vec_dum2)     = *(xp2+j) - *(xp0+j);
			*(vec_dum2 + 1) = *(yp2+j) - *(yp0+j);
			*(vec_dum2 + 2) = *(zp2+j) - *(zp0+j);

/* Calculate the local z basis vector for node j */
			check = normcrossX( vec_dum1, vec_dum2, local_z );
			if(!check) printf( "Problems with normcrossX \n");

			fdum1 = fabs(*(local_z));
			fdum2 = fabs(*(local_z+1));
			fdum3 = fabs(*(local_z+2));
/*
   The algorithm below is taken from "The Finite Element Method" by Thomas Hughes,
   page 388.  The goal is to find the local shell coordinates which come closest
   to the global x, y, z coordinates.  In the algorithm below, vec_dum is set to either
   the global x, y, or z basis vector based on the one of the 2 smaller components of the
   local z basis vector.  Once set, the cross product of the local z vector and vec_dum
   produces the local y direction.  This local y is then crossed with the local z to
   produce local x.
*/
			memset(vec_dum,0,nsd*sof);
			i2=1;
			if( fdum1 > fdum3)
			{
			    fdum3=fdum1;
			    i2=2;
			}
			if( fdum2 > fdum3) i2=3;
			*(vec_dum+(i2-1))=1.0;

/* Calculate the local y basis vector for node j */
			check = normcrossX( local_z, vec_dum, local_y );
			if(!check) printf( "Problems with normcrossX \n");

/* Calculate the local x basis vector for node j */
			check = normcrossX( local_y, local_z, local_x );
			if(!check) printf( "Problems with normcrossX \n");

			*(rotate + j*nsd2*nsd) = *(local_x);
			*(rotate + j*nsd2*nsd + 1) = *(local_x + 1);
			*(rotate + j*nsd2*nsd + 2) = *(local_x + 2);
			*(rotate + j*nsd2*nsd + 3) = *(local_y);
			*(rotate + j*nsd2*nsd + 4) = *(local_y + 1);
			*(rotate + j*nsd2*nsd + 5) = *(local_y + 2);

/* Put coord_el into local coordinates */

			check = matX( (coord_el_local+nsd2*j), (rotate + j*nsd2*nsd),
				(coord_el+nsd*j), nsd2, 1, nsd);
			if(!check) printf( "Problems with  matX \n");
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
		    }

/* The code below does the same as the matrix multiplication above to calculate the local
   coordinates and is clearer but isn't as efficient.  If the code below is used, make sure
   to comment out both the matrix multiplication above as well as the setting of
   coord_el_trans.

		    dum = nsd*npel;
		    check = rotXmat3(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat3 \n");
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
		    }
*/

		}

/* Assembly of the shg matrix for each integration point */

		check=qdshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with qdshg \n");

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(K_temp,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_local,0,neqlsq64*sof);

/* Assembly of the B matrix */

		    check = quadB((shg+npel*(nsd2+1)*j),B);
		    if(!check) printf( "Problems with quadB \n");

		    for( i1 = 0; i1 < neqel8; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel8*1+i1)*D12;
			*(DB+neqel8*1+i1) = *(B+i1)*D21+
				*(B+neqel8*1+i1)*D22;
			*(DB+neqel8*2+i1) = *(B+neqel8*2+i1)*G;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, B, DB, neqel8, neqel8, sdim);
		    if(!check) printf( "Problems with matXT  \n");
		    for( i2 = 0; i2 < neqlsq64; ++i2 )
		    {
			  *(K_el+i2) += *(K_local+i2)*wXdet;
		    }
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
				*(K_el + ndof2*neqel8*i + ndof2*j);
			   *(K_temp + ndof*neqel*i + ndof*j + 1) =
				*(K_el + ndof2*neqel8*i + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof*neqel*i + neqel + ndof*j) =
				*(K_el + ndof2*neqel8*i + neqel8 + ndof2*j);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(K_el + ndof2*neqel8*i + neqel8 + ndof2*j + 1);
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

		   check = matXrot2(K_temp, K_el, rotate, neqel8, neqel);
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


int qdConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to qdConjPassemble to get the
   product of [A]*[p].

                        Updated 8/22/06

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
		check = qdConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, p);
		if(!check) printf( " Problems with qdConjPassemble \n");
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
			*(U+j)*bet,*(r+j)*bet,*(P_global_CG+j)*alp/(*(mass+j)),
			*(force+j)*alp/(*(mass+j)));*/
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
	check = qdConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, U);
	if(!check) printf( " Problems with qdConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}



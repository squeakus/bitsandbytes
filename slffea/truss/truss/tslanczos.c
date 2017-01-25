/* This program calculates the eigenvalues of the matrix system:

                ([K] - lambda*[M])*[z] = [0]

        [K] = nXn  stiffness matrix
        [M] = nXn  mass matrix
        [z] = nX1  eigenvector
        lambda = eigenvalue

   using the Lanczos method.  It is taken from the algorithm given
   in table 10.6.1 in the book "The Finite Element Method" by Thomas Hughes, 
   page 588.  This is for a finite element program which does analysis on
   a truss.   Note that implicit in this algorithm is the assumption that [M]
   is positive definite.

                        Updated 11/7/06 

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
#include "../../common/eigen.h"

#define SMALL               1.0e-12
#define SMALLER             1.0e-20

extern int dof, neqn, sof;
extern int LU_decomp_flag, numel_K, numel_P;
extern int consistent_mass_flag, consistent_mass_store, eigen_print_flag,
	lumped_mass_flag;
extern int iteration_max;

int eval_print(EIGEN *, int );

int evector_print(double *, int , int );

int Lanczos_vec_print(double *, int , int );

int T_evector_print(double *, int , int );

int T_eval_print( double *, double *, int , int );

int T_print( double *, double *, int , int );

int compare_eigen ( const void *,  const void *);

int dotX(double *, double *, double *, int);

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int tsConjGrad( double *, BOUND , int *, double *, int *, double *, double *,
	double *, double *, MATL *, double *);

int solve(double *, double *, int *, int );

int tsMassPassemble( int *, double *, int *, double *, double *, double *,
	MATL *, double *, double *);

int QR_check( double *, double *, double *, int, int );

int tsLanczos(double *K, BOUND bc, int *connect, double *coord, EIGEN *eigen,
	int *el_matl, int *id, int *idiag, double *K_diag, double *length,
	double *local_xyz, double *M, MATL *matl, double *ritz, int num_eigen)
{
	int i,i2,j,jsave,k,check;
	int sofmf, ptr_inc;
	int *index;
	double *mem_double;
	double *r, *rbar, *rhat, *p, *pbar, *qm1, *Mq, *vv,
		*q, *T_evector, *T_diag, *T_upper, *temp;
	double *fdum_vector_in, *fdum_vector_out;
	double lamda, fdum, alpha, beta, err, d, dn;
	int n;

	n = neqn;

	index=(int *)malloc(n*sizeof(int));
	if(!index) printf( "error allocating memory for index\n");

	sofmf= num_eigen*num_eigen + 2*n*num_eigen + 5*n +
		num_eigen + num_eigen-1 + 2*dof;
	
	mem_double=(double *)calloc(sofmf,sizeof(double));
	if(!mem_double )
	{
		printf( "failed to allocate memory for double\n ");
		exit(1);
	}
	                                             ptr_inc=0;
	T_evector=(mem_double+ptr_inc);              ptr_inc += num_eigen*num_eigen;
	q=(mem_double+ptr_inc);                      ptr_inc += n*num_eigen;
	Mq=(mem_double+ptr_inc);                     ptr_inc += n*num_eigen;
	p=(mem_double+ptr_inc);                      ptr_inc += n;
	qm1=(mem_double+ptr_inc);                    ptr_inc += n;
	r=(mem_double+ptr_inc);                      ptr_inc += n;
	vv=(mem_double+ptr_inc);                     ptr_inc += n;
	temp=(mem_double+ptr_inc);                   ptr_inc += n;
	T_diag=(mem_double+ptr_inc);                 ptr_inc += num_eigen;
	T_upper=(mem_double+ptr_inc);                ptr_inc += num_eigen-1;
	fdum_vector_in=(mem_double+ptr_inc);         ptr_inc += dof;
	fdum_vector_out=(mem_double+ptr_inc);        ptr_inc += dof;

/* Initialize variables */

	for( i = 0; i < num_eigen; ++i )
	{
		eigen[i].index = i;
	}

	dn = 3.14159/((double) n);

	fdum = 0.0;
	for( i = 0; i < n; ++i ) 
	{
		*(qm1 + i) = 0.0;
		*(r + i) = sin(3.14159*fdum);
		fdum += dn;
	}
	/* *(r + 0) = 1.0;*/

/* Set p = M*r */

	if(lumped_mass_flag)
	{
	    for( i = 0; i < n; ++i ) 
	    {
		*(p + i) = *(M + i)*(*(r + i));
	    }
	}
	if(consistent_mass_flag)
	{

/* Finding [p]=[M][r] for when [M] is the consistent mass. 

   Note that the vector r has to be expanded so that
   the components which were removed based on fixed boundary
   conditions are restored.  
*/

	    memset(fdum_vector_in,0,dof*sof);
	    memset(fdum_vector_out,0,dof*sof);
	    for( j = 0; j < dof; ++j )
	    {
		if( *(id + j) > -1 )
		{
			*(fdum_vector_in + j) = *(r + *(id + j));
		}
	    }

	    check = tsMassPassemble( connect, coord, el_matl, length, local_xyz,
		M, matl, fdum_vector_out, fdum_vector_in);
	    if(!check) printf( " Problems with tsMassPassemble \n");

	    for( j = 0; j < dof; ++j )
	    {
		if( *(id + j) > -1 )
		{
			*(p + *(id + j)) = *(fdum_vector_out + j);
		}
	    }
	}

	check = dotX( &beta, r, p, n);
	if(!check) printf( " Problems with dotX \n");
	if( beta > 0.0 ) beta = sqrt(beta);
	else
	{
		printf(" beta = %16.8e < 0.0\n", beta);
		printf("Mass matrix may not be positive definite\n");
		exit(1);
	}

	for( i = 0; i < n; ++i ) 
	{
		*(q + i ) = *(r+i)/beta;

/* p = M*q = M*r/beta */

		*(p + i ) /= beta;
	}

	for( i = 0; i < n; ++i ) 
	{
		*(Mq + i) = *(p + i);
		*(r + i) = *(p + i);
	}

	err = 10000.0;

/* Begin iteration */

	/*while( err > 1.0e-6 )*/
	for( i = 0; i < num_eigen; ++i )
	{

	    if(LU_decomp_flag)
	    {

/* Using LU decomposition to solve the system */

		check = solve(K, r, idiag, n);
		if(!check) printf( " Problems with solve \n");
	    }

	    if(!LU_decomp_flag)
	    {

/* Using Conjugate gradient method to solve the system. 

   Note that the vector r has to be collasped and expanded so that
   the components which were removed based on fixed boundary
   conditions are restored. 
*/

		memset(fdum_vector_in,0,dof*sof);
		memset(fdum_vector_out,0,dof*sof);
		for( j = 0; j < dof; ++j )
		{
			if( *(id + j) > -1 )
			{
				*(fdum_vector_in + j) = *(r + *(id + j));
			}
		}

		check = tsConjGrad( K, bc, connect, coord, el_matl, fdum_vector_in,
			K_diag, length, local_xyz, matl, fdum_vector_out);
		if(!check) printf( " Problems with tsConjGrad \n");

		for( j = 0; j < dof; ++j )
		{
			if( *(id + j) > -1 )
			{
				*(r + *(id + j)) = *(fdum_vector_out + j);
			}
		}
	    }

	    for( j = 0; j < n; ++j )
	    {
		*(r + j) -= *(qm1 + j)*beta;
	    }

	    check = dotX( &alpha, p, r, n);
	    if(!check) printf( " Problems with dotX \n");

/* Cretate tridiagonal T matrix */

	    *(T_diag + i) = alpha;

	    if(i)
	    {
		*(T_upper + i - 1 ) = beta;
	    }

	    for( j = 0; j < n; ++j )
	    {
		*(r + j) -= *(q + n*i + j)*alpha;
	    }

	    if(lumped_mass_flag)
	    {
		for( j = 0; j < n; ++j ) 
		{
			*(p + j) = *(M + j)*(*(r + j));
		}
	    }
	    if(consistent_mass_flag)
	    {

/* Finding [p]=[M][r] for when [M] is the consistent mass.  */

		memset(fdum_vector_in,0,dof*sof);
		memset(fdum_vector_out,0,dof*sof);
		for( j = 0; j < dof; ++j )
		{
		    if( *(id + j) > -1 )
		    {
			*(fdum_vector_in + j) = *(r + *(id + j));
		    }
		}

		check = tsMassPassemble( connect, coord, el_matl, length, local_xyz,
			M, matl, fdum_vector_out, fdum_vector_in);
		if(!check) printf( " Problems with tsMassPassemble \n");

		for( j = 0; j < dof; ++j )
		{
		    if( *(id + j) > -1 )
		    {
			*(p + *(id + j)) = *(fdum_vector_out + j);
		    }
		}
	    }

	    check = dotX( &beta, p, r, n);
	    if(!check) printf( " Problems with dotX \n");
	    if( beta > 0.0 ) beta = sqrt(beta);
	    else
	    {
		printf(" beta = %16.8e < 0.0\n", beta);
		printf("Mass matrix may not be positive definite\n");
		exit(1);
	    }

	    memset(temp,0,n*sof);
	    if( i < num_eigen - 1)
	    {
		for( j = 0; j < n; ++j )
		{
			*(qm1 + j) = *(q + n*i + j);
			*(q + n*(i+1) + j) = *(r + j)/beta;
		}

/* Below is a re-orthogonalization step not given in the algorithm, but
   is needed to retain the orthonormal properties of q with Mq, which are
   lost due to the accumulation of numerical error.  So re-orthogonalize q
   using Gram-Schmidt
*/

		/* printf( "\n");*/
		for( k = 0; k < i+1; ++k )
		{
		    check = dotX( &fdum, (Mq + n*k), (q + n*(i+1)), n);
		    if(!check) printf( " Problems with dotX \n");
		    /*printf( "q(%d)M*q(%d) = %14.6e \n", k, i+1, fdum);*/
		    if ( fabs(fdum) > SMALL )
		    {
			for( i2 = 0; i2 < n; ++i2 )
			{
				*(temp + i2) += fdum*(*(q + n*k + i2));
				/*printf( " temp i2  %14.6e %5d \n", *(temp + i2), i2);*/
			}
		    }
		}
		for( j = 0; j < n; ++j )
		{
		    *(q + n*(i+1) + j) -= *(temp + j);
		}

/* Below is an extra normalization step not given in the algorithm, but
   is needed to retain the orthonormal properties of q with Mq.  So 
   calculate fdum = q*M*q and normalize q with q = q/sqrt(fdum)
*/

		if(lumped_mass_flag)
		{
		    for( j = 0; j < n; ++j ) 
		    {
			*(temp + j) = *(M + j)*(*(q + n*(i+1) + j));
		    }
		}
		if(consistent_mass_flag)
		{

/* Finding [temp]=[M][q + n*(i+1)] for when [M] is the consistent mass.  */

		    memset(fdum_vector_in,0,dof*sof);
		    memset(fdum_vector_out,0,dof*sof);
		    for( j = 0; j < dof; ++j )
		    {
			if( *(id + j) > -1 )
			{
			    *(fdum_vector_in + j) = *(q + n*(i+1) + *(id + j));
			}
		    }

		    check = tsMassPassemble( connect, coord, el_matl, length, local_xyz,
			M, matl, fdum_vector_out, fdum_vector_in);
		    if(!check) printf( " Problems with tsMassPassemble \n");

		    for( j = 0; j < dof; ++j )
		    {
			if( *(id + j) > -1 )
			{
			    *(temp + *(id + j)) = *(fdum_vector_out + j);
			}
		    }
		}

		check = dotX( &fdum, temp, (q + n*(i+1)), n);
		if(!check) printf( " Problems with dotX \n");
		if( fdum > 0.0 ) fdum = sqrt(fdum);
		else
		{
			printf(" fdum = %16.8e < 0.0\n", fdum);
			printf("Mass matrix may not be positive definite\n");
			exit(1);
		}

		for( j = 0; j < n; ++j )
		{
			*(q + n*(i+1) + j) /= fdum;
			*(p + j) =  *(temp + j)/fdum;
			*(Mq + n*(i+1) + j) = *(p + j);
			*(r + j) = *(p + j);
		}
	    }
	}

	if(eigen_print_flag)
	{
	    check = T_print( T_diag, T_upper, n, num_eigen);
	    if(!check) printf( " Problems with T_print \n");
	}

/* Calculate eigenvalues */

	check = QR_check(T_evector, T_diag, T_upper, iteration_max, num_eigen);
	if(!check) printf( " Problems with QR \n");

	if(eigen_print_flag)
	{
	    check = T_eval_print( T_diag, T_upper, n, num_eigen);
	    if(!check) printf( " Problems with T_eval_print \n");
	}

/* For inverse iteration, the eigenvalues of the original system relate
   to the computed system by 1/(T_diag).  See the paragraph on page 590
   below equation 10.6.24 of Hughes.
*/

	for( i = 0; i < num_eigen; ++i )
	{
		eigen[i].val = 1.0/(*(T_diag+i));
	}

	qsort( eigen, num_eigen, sizeof(EIGEN), compare_eigen);

	if(eigen_print_flag)
	{
	    check = T_evector_print(T_evector, n, num_eigen);
	    if(!check) printf( " Problems with T_evector_print \n");

	    check = Lanczos_vec_print(q, n, num_eigen);
	    if(!check) printf( " Problems with Lanczos_vec_print \n");
	}

	check = matXT(ritz, q, T_evector, n, num_eigen, num_eigen);
	if(!check) printf( " Problems with matXT \n");

	if(eigen_print_flag)
	{
	    printf("These are the eigenvectors of the original problem\n");
	    check = evector_print(ritz, n, num_eigen);
	    if(!check) printf( " Problems with evector_print \n");
	}

/* Redorder the eigenvectors */

	for( i = 0; i < n; ++i )
	{
	    for( j = 0; j < num_eigen; ++j )
	    {
		*(Mq + num_eigen*i + j) = *(ritz + num_eigen*i + eigen[j].index);
	    }
	}
	
	for( i = 0; i < n*num_eigen; ++i )
	{
		 *(ritz + i) = *(Mq + i);
	}

	if(eigen_print_flag)
	{
	    check = eval_print(eigen, num_eigen);
	    if(!check) printf( " Problems with eval_print \n");

	    printf("These are the eigenvectors in order\n");
	    check = evector_print(ritz, n, num_eigen);
	    if(!check) printf( " Problems with evector_print \n");
	}

	free(mem_double);
	free(index);

	return 1;
}


/* This program calculates the eigenvalues of the waveguide problem given
   in Section 8.2, page 244 - 247 of "The Finite Element Method in
   Electromagnetics" by Jianming Jin.  The system to solve is:

                ([K] - lambda*[M])*[z] = [0]

        [K] = nXn  Att matrix
        [M] = nXn  Btt' matrix = [Btz][Bzz]^(-1)[Bzt] - [Btt]
        [z] = nX1  eigenvector
        lambda = eigenvalue

   using the Lanczos method.  This implementation of the Lanczos method
   is taken from the algorithm given in table 10.6.1 in the book
   "The Finite Element Method" by Thomas Hughes, page 588.  This is for a
   finite element program which does analysis on a 4 node quadrilateral
   element.

                        Updated 6/20/02 

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
#include "../../common/eigen.h"

#define SMALL               1.0e-12
#define SMALLER             1.0e-20

extern int dof, EMdof, neqn, EMneqn, sof;
extern int LU_decomp_flag, numel_EM, numel_P;
extern int B_matrix_store, eigen_print_flag;
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

#if 0
int qd3ConjGrad(double *, BOUND , int *, double *, int *, double *, double *, MATL *,
	double *);
#endif

int solve(double *, double *, int *, int );

int qdBPassemble(int *, double *, int *, double *, MATL *, double *, double *);

int QR_check( double *, double *, double *, int, int );

int qd3Lanczos(double *K, double *Btt, double *Btz, double *Bzt, double *Bzz, BOUND bc,
	int *connect, double *coord, EIGEN *eigen, int *el_matl, int *id, int *idiag,
	double *Att_diag,  MATL *matl, double *ritz, int num_eigen)
{
	int i,i2,j,jsave,k,check;
        int sofmf, sofeigen, ptr_inc;
        int *index;
	double *mem_double;
	double *r, *rbar, *rhat, *p, *pbar, *qm1, *Mq, *vv,
		*q, *T_evector, *T_diag, *T_upper, *temp, *temp2;
	double *fdum_vector_in, *fdum_vector_out;
	double lamda, fdum, alpha, beta, err, d, dn;
	int n;

	n = EMneqn;

        index=(int *)malloc(n*sizeof(int));
        if(!index) printf( "error allocating memory for index\n");

	sofmf= num_eigen*num_eigen + 2*n*num_eigen + 4*n + 2*EMdof + 
		num_eigen + num_eigen-1 + 2*EMdof;
	
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
	temp=(mem_double+ptr_inc);                   ptr_inc += EMdof;
	temp2=(mem_double+ptr_inc);                  ptr_inc += EMdof;
	T_diag=(mem_double+ptr_inc);                 ptr_inc += num_eigen;
	T_upper=(mem_double+ptr_inc);                ptr_inc += num_eigen-1;
       	fdum_vector_in=(mem_double+ptr_inc);         ptr_inc += EMdof;
       	fdum_vector_out=(mem_double+ptr_inc);        ptr_inc += EMdof;

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

/* Finding [p]=[M][r] for when [M] = [Btz][Bzz]^(-1)[Bzt] - [Btt]. 
*/


	memset(fdum_vector_in,0,EMdof*sof);
	memset(fdum_vector_out,0,EMdof*sof);
	memset(temp,0,EMdof*sof);
	memset(temp2,0,EMdof*sof);

/* 1st, find [temp] = [Btt][r]

   Note that the vector r has to be expanded so that
   the components which were removed based on fixed boundary
   conditions are restored.  
*/

	for( j = 0; j < EMdof; ++j )
	{
		if( *(id + j) > -1 )
		{
			*(fdum_vector_in + j) = *(r + *(id + j));
			printf("fffff %4d %12.8f \n",*(id + j), *(fdum_vector_in + j));
		}
	}

	check = qdBPassemble(connect, coord, el_matl, Btt, matl,
		fdum_vector_out, fdum_vector_in);
        if(!check) printf( " Problems with qdBPassemble \n");

	for( j = 0; j < EMdof; ++j )
	{
		if( *(id + j) > -1 )
		{
			*(temp + *(id + j)) = *(fdum_vector_out + j);
			printf("ftftft  %4d %12.8f \n",*(id + j), *(temp + *(id + j)));
		}
	}

/* 2nd, find [temp2] = [Bzt][r] */

	memset(fdum_vector_out,0,EMdof*sof);
	check = qdBPassemble(connect, coord, el_matl, Bzt, matl,
		fdum_vector_out, fdum_vector_in);
        if(!check) printf( " Problems with qdBPassemble \n");

	for( j = 0; j < EMdof; ++j )
	{
		if( *(id + j) > -1 )
		{
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
			printf("ttttt %4d %12.8f \n",*(id + j), *(temp2 + *(id + j)));
		}
	}

/* 3rd, find [temp2] = [Bzz]^(-1)[temp2] = [Bzz]^(-1)[Bzt][r] */

	check = solve(Bzz, temp2, idiag, n);
	if(!check) printf( " Problems with solve \n");

/* 4th, find [temp2] = [Btz][temp2] = [Btz][Bzz]^(-1)[Bzt][r] */

	memset(fdum_vector_in,0,EMdof*sof);
	memset(fdum_vector_out,0,EMdof*sof);
	for( j = 0; j < EMdof; ++j )
	{
		if( *(id + j) > -1 )
		{
			*(fdum_vector_in + j) = *(temp2 + *(id + j));
			printf("tvtvtv %4d %12.8f \n",*(id + j), *(fdum_vector_in + j));
		}
	}

	check = qdBPassemble(connect, coord, el_matl, Btz, matl,
		fdum_vector_out, fdum_vector_in);
        if(!check) printf( " Problems with qdBPassemble \n");

	for( j = 0; j < EMdof; ++j )
	{
		if( *(id + j) > -1 )
		{
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
			printf("rrrrr %4d %12.8f \n",*(id + j), *(temp2 + *(id + j)));
		}
	}

/* 5th, find [p] = [temp2] - [temp] = [Btz][Bzz]^(-1)[Bzt][r] - [Btt][r] */

	for( j = 0; j < n; ++j )
	{
		*(p + j) = *(temp2 + j) - *(temp + j);
		printf("ppppp %4d %12.8f  %12.8f \n", j, *(p + j), *(r + j));
	}

	check = dotX( &beta, r, p, n);
        if(!check) printf( " Problems with dotX \n");
	beta = fabs(beta);
	beta = sqrt(beta);
#if 0
	if( beta > 0.0 ) beta = sqrt(beta);
	else
	{
		printf(" beta = %16.8e < 0.0\n", beta);
		printf("Mass matrix may not be positive definite\n", beta);
		exit(1);
	}
#endif

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

#if 0
	    if(!LU_decomp_flag)
	    {

/* Using Conjugate gradient method to solve the system. 

   Note that the vector r has to be collasped and expanded so that
   the components which were removed based on fixed boundary
   conditions are restored. 
*/

	    	memset(fdum_vector_in,0,EMdof*sof);
	    	memset(fdum_vector_out,0,EMdof*sof);
		for( j = 0; j < EMdof; ++j )
		{
			if( *(id + j) > -1 )
			{
				*(fdum_vector_in + j) = *(r + *(id + j));
			}
		}

		check = qdConjGrad( K, bc, connect, coord, el_matl, fdum_vector_in,
			Att_diag, matl, fdum_vector_out);
		if(!check) printf( " Problems with qdConjGrad \n");

		for( j = 0; j < EMdof; ++j )
		{
			if( *(id + j) > -1 )
			{
				*(r + *(id + j)) = *(fdum_vector_out + j);
			}
		}
	    }
#endif

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

/* Finding [p]=[M][r] for when [M] = [Btz][Bzz]^(-1)[Bzt] - [Btt].  */

	    memset(fdum_vector_in,0,EMdof*sof);
	    memset(fdum_vector_out,0,EMdof*sof);
	    memset(temp,0,EMdof*sof);
	    memset(temp2,0,EMdof*sof);

/* 1st, find [temp] = [Btt][r] */

	    for( j = 0; j < EMdof; ++j )
	    {
		    if( *(id + j) > -1 )
		    {
			*(fdum_vector_in + j) = *(r + *(id + j));
		    }
	    }

	    check = qdBPassemble(connect, coord, el_matl, Btt, matl,
		    fdum_vector_out, fdum_vector_in);
            if(!check) printf( " Problems with qdBPassemble \n");

	    for( j = 0; j < EMdof; ++j )
	    {
		    if( *(id + j) > -1 )
		    {
			*(temp + *(id + j)) = *(fdum_vector_out + j);
		    }
	    }

/* 2nd, find [temp2] = [Bzt][r] */

	    memset(fdum_vector_out,0,EMdof*sof);
	    check = qdBPassemble(connect, coord, el_matl, Bzt, matl,
                fdum_vector_out, fdum_vector_in);
	    if(!check) printf( " Problems with qdBPassemble \n");

	    for( j = 0; j < EMdof; ++j )
	    {
		if( *(id + j) > -1 )
		{
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
		}
	    }

/* 3rd, find [temp2] = [Bzz]^(-1)[temp2] = [Bzz]^(-1)[Bzt][r] */

	    check = solve(Bzz, temp2, idiag, n);
	    if(!check) printf( " Problems with solve \n");

/* 4th, find [temp2] = [Btz][temp2] = [Btz][Bzz]^(-1)[Bzt][r] */

	    memset(fdum_vector_in,0,EMdof*sof);
	    memset(fdum_vector_out,0,EMdof*sof);
	    for( j = 0; j < EMdof; ++j )
	    {
		if( *(id + j) > -1 )
		{
			*(fdum_vector_in + j) = *(temp2 + *(id + j));
		}
	    }

	    check = qdBPassemble(connect, coord, el_matl, Btz, matl,
		fdum_vector_out, fdum_vector_in);
	    if(!check) printf( " Problems with qdBPassemble \n");

	    for( j = 0; j < EMdof; ++j )
	    {
		if( *(id + j) > -1 )
		{
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
		}
	    }

/* 5th, find [p] = [temp2] - [temp] = [Btz][Bzz]^(-1)[Bzt][r] - [Btt][r] */

	    for( j = 0; j < n; ++j )
	    {
		*(p + j) = *(temp2 + j) - *(temp + j);
	    }

	    check = dotX( &beta, p, r, n);
            if(!check) printf( " Problems with dotX \n");
	    beta = fabs(beta);
	    beta = sqrt(beta);
#if 0
	    if( beta > 0.0 ) beta = sqrt(beta);
	    else
	    {
		printf(" beta = %16.8e < 0.0\n", beta);
		printf("Mass matrix may not be positive definite\n", beta);
		exit(1);
	    }
#endif

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

/* Finding [temp]=[M][q + n*(i+1)] for when [M] = [Btz][Bzz]^(-1)[Bzt] - [Btt].  */

		memset(fdum_vector_in,0,EMdof*sof);
		memset(fdum_vector_out,0,EMdof*sof);
		memset(temp,0,EMdof*sof);
		memset(temp2,0,EMdof*sof);

/* 1st, find [temp] = [Btt][r] */

		for( j = 0; j < EMdof; ++j )
		{
			if( *(id + j) > -1 )
			{
			    *(fdum_vector_in + j) = *(q + n*(i+1) + *(id + j));
			}
		}

		check = qdBPassemble(connect, coord, el_matl, Btt, matl,
		    	fdum_vector_out, fdum_vector_in );
		if(!check) printf( " Problems with qdBPassemble \n");

		for( j = 0; j < EMdof; ++j )
		{
			if( *(id + j) > -1 )
			{
			    *(temp + *(id + j)) = *(fdum_vector_out + j);
			}
		}

/* 2nd, find [temp2] = [Bzt][r] */

		memset(fdum_vector_out,0,EMdof*sof);
		check = qdBPassemble(connect, coord, el_matl, Bzt, matl,
			fdum_vector_out, fdum_vector_in);
		if(!check) printf( " Problems with qdBPassemble \n");

		for( j = 0; j < EMdof; ++j )
		{
		    if( *(id + j) > -1 )
		    {
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
		    }
		}

/* 3rd, find [temp2] = [Bzz]^(-1)[temp2] = [Bzz]^(-1)[Bzt][r] */

		check = solve(Bzz, temp2, idiag, n);

/* 4th, find [temp2] = [Btz][temp2] = [Btz][Bzz]^(-1)[Bzt][r] */

		memset(fdum_vector_in,0,EMdof*sof);
		memset(fdum_vector_out,0,EMdof*sof);
		for( j = 0; j < EMdof; ++j )
		{
		    if( *(id + j) > -1 )
		    {
			*(fdum_vector_in + j) = *(temp2 + *(id + j));
		    }
		}

		check = qdBPassemble(connect, coord, el_matl, Btz, matl,
			fdum_vector_out, fdum_vector_in);
		if(!check) printf( " Problems with qdBPassemble \n");

		for( j = 0; j < EMdof; ++j )
		{
		    if( *(id + j) > -1 )
		    {
			*(temp2 + *(id + j)) = *(fdum_vector_out + j);
		    }
		}

/* 5th, find [temp] = [temp2] - [temp] = [Btz][Bzz]^(-1)[Bzt][r] - [Btt][r] */

		for( j = 0; j < n; ++j )
		{
		    *(temp + j) = *(temp2 + j) - *(temp + j);
		}

                check = dotX( &fdum, temp, (q + n*(i+1)), n);
            	if(!check) printf( " Problems with dotX \n");
		fdum = fabs(fdum);
		fdum = sqrt(fdum);
#if 0
		if( fdum > 0.0 ) fdum = sqrt(fdum);
		else
		{
			printf(" fdum = %16.8e < 0.0\n", fdum);
			printf("Mass matrix may not be positive definite\n", fdum);
			exit(1);
		}
#endif

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

	return 1;
}


/*
    This subroutine performs skyline decomposition and also calculates
    the solution of the skylined system.  It is based on Dr. David Benson's 
    algorithm for ames 232 fall 1996.  

	Implemented by San Le.

                Updated 9/30/99 

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

int dotX(double *,double *, double *, int );

/* This program performs back substitution on a skylined linear
   system */

int solve(double *A,double *f,int *idiag, int neq)
{
	int i,j, k, iloci, ri, rj, ipntr, len, check;
	double p; 


/* forward substitution */

	for( i = 1; i < neq; ++i )
	{
/* Calculate the fist row in column k */
		ri = i + 1 + *(idiag + i - 1 ) - *(idiag + i);
		if( ri <= i - 1)
		{
			iloci = *(idiag + i) - (i - ri);
			len = i - ri;
			check = dotX(&p,(A+iloci),(f+ri),len);
			*(f + i) -= p;
		}
	}
/* diagonal scaling */

	for( i = 0; i < neq; ++i )
	{
		if( *(A + *(idiag + i)) != 0)
		{
			*(f + i) /= *(A + *(idiag + i));
		}
	}
/* Backward substitution */ 
	for( j = neq-1; j > 0; --j )
	{
/* Calculate the fist row in column j */
		rj = j + 1 + *(idiag + j - 1 ) - *(idiag + j);
		if( rj <= j - 1)
		{
			ipntr = *(idiag + j) - j;
			for( i = rj; i < j ; ++i )
			{
				*(f + i) -= *(f + j)*(*(A + ipntr + i));
			}
		}
	}
	return 1;
}


/* This program performs LU decomposition on a skylined matrix */
int decomp(double *A,int *idiag,int neq)
{
	int i,j,k,rk,ri, check;
	int iloci, ilock, iloc, len;
	double p, t;

	for( k = 1; k < neq; ++k )
	{
/* Calculate the fist row in column k */

		rk = k + 1 + *(idiag + k - 1 ) - *(idiag + k);
		
		if( rk + 1 <=  k - 1)
		{
			for( i = rk + 1; i < k ; ++i )
			{
/* Calculate the fist row in column i */
			   ri = i + 1 + *(idiag + i - 1 ) - *(idiag + i);
/* Calculate where the overlap begins  */
			   j = MAX(ri,rk);
			   iloci = *(idiag + i) - (i-j);
			   ilock = *(idiag + k) - (k-j);
			   len = i - j;
			   check = dotX(&p,(A+iloci),(A+ilock),len);
/* Calculate yi in column k  */
			   iloc = *(idiag + k) - (k - i);
			   *(A + iloc) = *(A + iloc) - p;
			}
		}
/* Calculate dk  */
		if( rk <= k - 1)
		{
			for( i = rk; i < k ; ++i)
			{
/* Calculate contribution to u(transpose)Du  */
			   iloc = *(idiag + k) - (k - i);
			   t = *(A + iloc);
			   *(A + iloc) = t/(*(A + *(idiag + i)));
			   *(A + *(idiag + k)) -= t*(*(A + iloc));
			}
		}
	}
	return 1;
}


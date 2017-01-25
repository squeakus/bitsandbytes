/* This program calculates the eigenvalues of a tridiagonal
   matrix using the QR with Wilkinson shift.  It is taken from
   the algorithms 5.1.3, 8.3.2 and 8.3.3 given in "Matrix Computations",
   by Golub, page 216, 420, 421. 

   I have made a significant amount of optimization to take
   advantage of the tridiagonal stucture of T.

                        Updated 9/2/00  

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

#define BIG                 1.0e+12
#define SMALL               1.0e-12
#define SMALLER             1.0e-20

int matXT(double *, double *, double *, int, int, int);

int givens( double *a, double *b)
{
/* This is algorithm 5.1.3
*/
	double tau, c, s, fb, fa;

	fb = fabs(*b);
	fa = fabs(*a);
	
	if( fb < SMALLER)
	{
		c = 1.0;
		s = 0.0;
	}
	else
	{
		tau = -*b/(*a);
		c = 1.0/sqrt(1.0 + tau*tau);
		s = tau*c; 
		if (  fb > fa);
		{
			tau = - *a/(*b);
			s = 1.0/sqrt(1.0 + tau*tau);
			c = s*tau;
		}
	}
	*a = c;
	*b = s;

	return 1;
}

int QR2X2( double *evector, double *T_diag, double *T_upper,
	int num_eigen, int n2, int shrink2)
{
/* This is algorithm 8.3.2 for a 2X2 matrix T
*/
	int i,j,jsave,k,check;
	double c, s, d, factor, mu, T_diag_n2m1, T_diag_n2m2, T_upper_n2m2;
	double sub_T[4], temp[4], G[4], fdum, fdum2;

	T_diag_n2m1 = *(T_diag + n2 - 1); 
	T_diag_n2m2 = *(T_diag + n2 - 2); 
	T_upper_n2m2 = *(T_upper + n2 - 2); 

	d = (T_diag_n2m2 - T_diag_n2m1)/2.0;

	factor = 1.0;
	if(d < 0) factor = -1.0;

	mu = T_diag_n2m1 -
	   T_upper_n2m2*T_upper_n2m2/(d + factor*sqrt(d*d + T_upper_n2m2*T_upper_n2m2));

	c = *(T_diag) - mu;
	s = *(T_upper);

	check = givens(&c,&s);
	if(!check) printf( " Problems with givens \n");

/*
           | T_diag    T_upper     |
sub_T  =   | T_upper   T_diag + 1  |

   G   =   |   c    s  |
           |  -s    c  |
*/

	*(G) = c;
	*(G + 1) = s;
	*(G + 2) = -s;
	*(G + 3) = c;

	*(sub_T ) = *(T_diag);
	*(sub_T + 1) = *(T_upper);
	*(sub_T + 2) = *(T_upper);
	*(sub_T + 3) = *(T_diag + 1);

/* Apply rotation to T: G(0)(transpose)*T*(G(0)) matrix */

/* G(transpose)*T */

	check = matXT(temp, G, sub_T, 2, 2, 2);
	if(!check) printf( " Problems with matXT \n");

/* T*G */

	*(sub_T) = *(temp)*c - *(temp+1)*s;
	*(sub_T + 1) = *(temp)*s + *(temp+1)*c;
	*(sub_T + 2) = *(temp+2)*c - *(temp+3)*s;
	*(sub_T + 3) = *(temp+2)*s + *(temp+3)*c;

	*(T_diag) = *(sub_T);
	*(T_diag + 1) = *(sub_T + 3);

	*(T_upper) = *(sub_T + 1);

/* Calculate eigenvectors */

	for( i = 0; i < num_eigen; ++i )
	{
		fdum = *(evector + num_eigen*i + shrink2)*c -
			*(evector + num_eigen*i + shrink2 + 1)*s;
		fdum2 = *(evector + num_eigen*i + shrink2)*s +
			*(evector + num_eigen*i + shrink2 + 1)*c;

		*(evector + num_eigen*i + shrink2) = fdum;
		*(evector + num_eigen*i + shrink2 + 1) = fdum2;
	}

	return 1;
}


int QR( double *evector, double *T_diag, double *T_upper,
	int num_eigen, int n2, int shrink2)
{
/* This is algorithm 8.3.2
*/
	int i,j,jsave,k,check;
	double c, s, d, factor, mu, T_diag_n2m1, T_diag_n2m2, T_upper_n2m2;
	double sub_T[16], nonzero, temp[16], G[4], fdum, fdum2;

	T_diag_n2m1 = *(T_diag + n2 - 1); 
	T_diag_n2m2 = *(T_diag + n2 - 2); 
	T_upper_n2m2 = *(T_upper + n2 - 2); 

	d = (T_diag_n2m2 - T_diag_n2m1)/2.0;

	factor = 1.0;
	if(d < 0) factor = -1.0;

	mu = T_diag_n2m1 -
	   T_upper_n2m2*T_upper_n2m2/(d + factor*sqrt(d*d + T_upper_n2m2*T_upper_n2m2));

	c = *(T_diag) - mu;
	s = *(T_upper);

	check = givens(&c,&s);
	if(!check) printf( " Problems with givens \n");

	nonzero = 0.0;
/*
	printf( "sine cosine\n ");
	printf( "%8.6e %8.6e\n ",s,c);
*/

/*
           | T_diag    T_upper        0.0          |
sub_T  =   | T_upper   T_diag + 1     T_upper + 1  |
           | 0.0       T_upper + 1    T_diag + 2   |

   G   =   |   c    s  |
           |  -s    c  |
*/

	*(G) = c;
	*(G + 1) = s;
	*(G + 2) = -s;
	*(G + 3) = c;

	*(sub_T ) = *(T_diag);
	*(sub_T + 1) = *(T_upper);
	*(sub_T + 2) = 0.0;
	*(sub_T + 3) = *(T_upper);
	*(sub_T + 4) = *(T_diag + 1);
	*(sub_T + 5) = *(T_upper + 1);
	*(sub_T + 6) = 0.0;
	*(sub_T + 7) = *(T_upper + 1);
	*(sub_T + 8) = *(T_diag + 2);

	*(temp + 6) = *(sub_T + 6);
	*(temp + 7) = *(sub_T + 7);
	*(temp + 8) = *(sub_T + 8);

/* Apply rotation to T: G(0)(transpose)*T*(G(0)) matrix */

/* G(transpose)*T */

	check = matXT(temp, G, sub_T, 2, 3, 2);
	if(!check) printf( " Problems with matXT \n");

/* T*G */

	*(sub_T) = *(temp)*c - *(temp+1)*s;
	*(sub_T + 1) = *(temp)*s + *(temp+1)*c;
	*(sub_T + 2) = *(temp + 2);
	*(sub_T + 3) = *(temp+3)*c - *(temp+4)*s;
	*(sub_T + 4) = *(temp+3)*s + *(temp+4)*c;
	*(sub_T + 5) = *(temp+5);
	*(sub_T + 6) = *(temp+6)*c - *(temp+7)*s;
	*(sub_T + 7) = *(temp+6)*s + *(temp+7)*c;
	*(sub_T + 8) = *(temp+8);

	*(T_diag) = *(sub_T);
	*(T_diag + 1) = *(sub_T + 4);
	*(T_diag + 2) = *(sub_T + 8);

	*(T_upper) = *(sub_T + 1);
	*(T_upper + 1) = *(sub_T + 5);

	nonzero = *(sub_T + 2);

/* Calculate eigenvectors */

	for( i = 0; i < num_eigen; ++i )
	{
		fdum = *(evector + num_eigen*i + shrink2)*c -
			*(evector + num_eigen*i + shrink2 + 1)*s;
		fdum2 = *(evector + num_eigen*i + shrink2)*s +
			*(evector + num_eigen*i + shrink2 + 1)*c;

		*(evector + num_eigen*i + shrink2) = fdum;
		*(evector + num_eigen*i + shrink2 + 1) = fdum2;
	}

	for( k = 0; k < n2-3; ++k )
	{
	    c = *(T_upper + k );
	    s = nonzero;

	    check = givens(&c,&s);
	    if(!check) printf( " Problems with givens \n");

/*
           | T_diag + k   T_upper + k        nonzero            0.0             |
sub_T  =   | T_upper + k  T_diag + k + 1     T_upper + k + 1    0.0             |
           | nonzero      T_upper + k + 1    T_diag + k + 2     T_upper + k + 2 |
           | 0.0          0.0                T_upper + k + 2    T_diag + k + 3  |

   G   =   |   c    s  |
           |  -s    c  |
*/

/* Calculate rotation angles */

	    *(G) = c;
	    *(G + 1) = s;
	    *(G + 2) = -s;
	    *(G + 3) = c;

/* Apply rotation to T: G(k)(transpose)*T*(G(k)) matrix */

/* G(transpose)*T */

	    *(sub_T ) = *(T_diag + k);
	    *(sub_T + 1) = *(T_upper + k);
	    *(sub_T + 2) = nonzero ;
	    *(sub_T + 3) = 0.0;
	    *(sub_T + 4) = *(T_upper + k);
	    *(sub_T + 5) = *(T_diag + k + 1);
	    *(sub_T + 6) = *(T_upper + k + 1);
	    *(sub_T + 7) = 0.0;
	    *(sub_T + 8) = nonzero;
	    *(sub_T + 9) = *(T_upper + k + 1);
	    *(sub_T + 10) = *(T_diag + k + 2);
	    *(sub_T + 11) = *(T_upper + k + 2);
	    *(sub_T + 12) = 0.0;
	    *(sub_T + 13) = 0.0;
	    *(sub_T + 14) = *(T_upper + k + 2);
	    *(sub_T + 15) = *(T_diag + k + 3);

	    *(temp) = *(sub_T);
	    *(temp + 1) = *(sub_T + 1);
	    *(temp + 2) = *(sub_T + 2);
	    *(temp + 3) = *(sub_T + 3);

	    *(temp + 12) = *(sub_T + 12);
	    *(temp + 13) = *(sub_T + 13);
	    *(temp + 14) = *(sub_T + 14);
	    *(temp + 15) = *(sub_T + 15);

	    check = matXT((temp+4), G, (sub_T+4), 2, 4, 2);
	    if(!check) printf( " Problems with matXT \n");

/* T*G */

	    *(sub_T + 1) = *(temp+1)*c - *(temp+2)*s;
	    *(sub_T + 2) = 0.0;
	    *(sub_T + 3) = 0.0;
	    *(sub_T + 4) = *(temp+4);
	    *(sub_T + 5) = *(temp+5)*c - *(temp+6)*s;
	    *(sub_T + 6) = *(temp+5)*s + *(temp+6)*c;
	    *(sub_T + 7) = *(temp+7);
	    *(sub_T + 8) = 0.0;
	    *(sub_T + 9) = *(temp+9)*c - *(temp+10)*s;
	    *(sub_T + 10) = *(temp+9)*s + *(temp+10)*c;
	    *(sub_T + 11) = *(temp+11);
	    *(sub_T + 12) = 0.0;
	    *(sub_T + 13) =  -*(temp+14)*s;
	    *(sub_T + 14) = *(temp+14)*c;

	    nonzero = *(sub_T + 7);

	    *(T_diag + k) = *(sub_T);
	    *(T_diag + k + 1) = *(sub_T + 5);
	    *(T_diag + k + 2) = *(sub_T + 10);
	    *(T_diag + k + 3) = *(sub_T + 15);

	    *(T_upper + k) = *(sub_T + 1);
	    *(T_upper + k + 1) = *(sub_T + 6);
	    *(T_upper + k + 2) = *(sub_T + 11);

/* Calculate eigenvectors */

	    for( i = 0; i < num_eigen; ++i )
	    {
		fdum = *(evector + num_eigen*i + shrink2 + k+1)*c -
			*(evector + num_eigen*i + shrink2 + k+2)*s;
		fdum2 = *(evector + num_eigen*i + shrink2 + k+1)*s +
			*(evector + num_eigen*i + shrink2 + k+2)*c;

		*(evector + num_eigen*i + shrink2 + k+1) = fdum;
		*(evector + num_eigen*i + shrink2 + k+2) = fdum2;
	    }
	}

	c = *(T_upper + n2 - 3 );
	s = nonzero;

	check = givens(&c,&s);
	if(!check) printf( " Problems with givens \n");

/*
           | T_diag + n2 - 3    T_upper + n2 - 3    nonzero           |
sub_T  =   | T_upper + n2 - 3   T_diag + n2 - 2     T_upper + n2 - 2  |
           | nonzero            T_upper + n2 - 2    T_diag + n2 - 1   |

   G   =   |   c    s  |
           |  -s    c  |
*/

/* Calculate last G(n2-2)(transpose)*T*(G(n2-2)) */

	*(G) = c;
	*(G + 1) = s;
	*(G + 2) = -s;
	*(G + 3) = c;

	*(sub_T ) = *(T_diag + n2 - 3);
	*(sub_T + 1) = *(T_upper + n2 - 3);
	*(sub_T + 2) = nonzero;
	*(sub_T + 3) = *(T_upper + n2 - 3);
	*(sub_T + 4) = *(T_diag + n2 - 2);
	*(sub_T + 5) = *(T_upper + n2 -2);
	*(sub_T + 6) = nonzero;
	*(sub_T + 7) = *(T_upper + n2 - 2);
	*(sub_T + 8) = *(T_diag + n2 - 1);

	*(temp) = *(sub_T);
	*(temp + 1) = *(sub_T + 1);
	*(temp + 2) = *(sub_T + 2);

/* Apply rotation to T: G(n2-3)(transpose)*T*G(n2-3) matrix */

/* G(transpose)*T */

	check = matXT((temp+3), G, (sub_T+3), 2, 3, 2);
	if(!check) printf( " Problems with matXT \n");

/* T*G */

	*(sub_T) = *(temp);
	*(sub_T + 1) = *(temp+1)*c - *(temp+2)*s;
	*(sub_T + 2) = *(temp+1)*s + *(temp+2)*c;
	*(sub_T + 3) = *(temp+3);
	*(sub_T + 4) = *(temp+4)*c - *(temp+5)*s;
	*(sub_T + 5) = *(temp+4)*s + *(temp+5)*c;
	*(sub_T + 6) = *(temp+6);
	*(sub_T + 7) = *(temp+7)*c - *(temp+8)*s;
	*(sub_T + 8) = *(temp+7)*s + *(temp+8)*c;

	*(T_diag + n2 - 3) = *(sub_T);
	*(T_diag + n2 - 2) = *(sub_T + 4);
	*(T_diag + n2 - 1) = *(sub_T + 8);

	*(T_upper + n2 - 3) = *(sub_T + 1);
	*(T_upper + n2 - 2) = *(sub_T + 5);

/* Calculate eigenvectors */

	if( n2 > 2)
	{
		for( i = 0; i < num_eigen; ++i )
		{
		    fdum = *(evector + num_eigen*i + shrink2 + n2-2)*c -
			*(evector + num_eigen*i + shrink2 + n2-1)*s;
		    fdum2 = *(evector + num_eigen*i + shrink2 + n2-2)*s +
			*(evector + num_eigen*i + shrink2 + n2-1)*c;

		    *(evector + num_eigen*i + shrink2 + n2-2) = fdum;
		    *(evector + num_eigen*i + shrink2 + n2-1) = fdum2;
		}
	}

	return 1;
}

int QR_check(double *evector, double *T_diag, double *T_upper, int iteration_max,
	int num_eigen)
{
	int i, j, k, dum, n2, check, counter, shrink, shrink2;
	double fdum;

#if 0
	*(T_diag) = 1.0;
	*(T_diag + 1) = 2.0;
	*(T_diag + 2) = 3.0;
	*(T_diag + 3) = 4.0;
	*(T_upper ) = 1.0;
	*(T_upper + 1) = 1.0;
	*(T_upper + 2) = .01;

	*(T_diag) = 1.0;
	*(T_diag + 1) = 3.0;
	*(T_diag + 2) = 5.0;
	*(T_diag + 3) = 7.0;

	*(T_upper ) = 2.0;
	*(T_upper + 1) = 4.0;
	*(T_upper + 2) = 6.0;
#endif

	for( i = 0; i < num_eigen; ++i )
	{
		*(evector + num_eigen*i + i) = 1.0;
	}

/* This is the implementation of Algorithm 8.3.3.  It sets off diagonal
   elements to zero, and sends a reduced T to QR.

   The variable "shrink" represents the dimension which the T matrix shrinks from
   the right.  In terms of the matrix D given on page 421:

      shrink = n - q
   
   where q represents the part of T to the right which has been rendered diagonal.
   The variable "shrink2" is the dimension which the T matrix shrinks from
   the left.  In terms of the matrix D given on page 421:

      shrink2 = p
   
   where p represents the part of T to the left which has been rendered diagonal.
*/
	counter = 0;
	shrink = num_eigen - 2;
	shrink2 = 0;
	n2 = num_eigen;
	while( n2 > 1 )
	{
		if( n2 > 2)
		{
		    check = QR(evector, (T_diag + shrink2), (T_upper + shrink2),
			num_eigen, n2, shrink2);
		}
		else
		{
/* For n2 = 2 */
		    check = QR2X2(evector, (T_diag + shrink2), (T_upper + shrink2),
			num_eigen, n2, shrink2);
		}

		fdum = fabs(*( T_upper + shrink ));
		while( fdum < SMALL )
		{
			*( T_upper + shrink ) = 0.0;
			--shrink;
			fdum = fabs(*( T_upper + shrink ));
			if (shrink < 1) fdum = BIG;
		}
		n2 = shrink + 2;
		fdum = fabs(*( T_upper + shrink2 ));
		while( fdum < SMALL )
		{
			*( T_upper + shrink2 ) = 0.0;
			++shrink2;
			fdum = fabs(*( T_upper + shrink2 ));
			if (shrink2 > n2-1) fdum = BIG;
		}
		n2 -= shrink2;
		++counter;
		if (counter > iteration_max)
		{
		    printf( "\nMaximum iterations %6d reached.  n2 is: %d\n",counter,n2);
		    printf( "Problem may not have converged during QR algorithm.\n");
		    n2 = 0;
		}
	}
	printf( "\n\n counter %d %d\n\n",counter, n2);

	return 1;
}

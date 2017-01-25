/*
    This utility function calculates the roots of a cubic equation.
    It reads in the coefficients of the cubic and replaces them
    with the roots of the cubic. 

    I am using the trig formula as given by Numerical Recipes in C, 2nd ed.
    page 184-185;

		Updated 5/3/02

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

#define THIRD      .333333333 
#define PI        3.141592654 
#define SMALL     1.0e-8
#define SMALLER   1.0e-14
#define SMALLEST  1.0e-44
#define BIG       1.0e10

int cubic(double *coef)
{
	int i, j, k;
	double a, b, c, Q, Q3, rt2Q, R, theta, maxroot, midroot, minroot,
		b2, c2, quad, signb, power, scale, scalesq,
		maga, magb, magc, fdum, fdum1, fdum2, fdum3, ratio;

	fdum = fabs(*(coef)) + fabs(*(coef+1)) + fabs(*(coef+2)); 

	if( fdum < SMALLER )
	{
		*(coef) = 0.0;
		*(coef+1) = 0.0;
		*(coef+2) = 0.0;
		return 1;
	}

/* Scale equations for case of very large or small coefficients */

	fdum = fabs(*(coef)) + SMALLER; 

	power = (double)((int)(log10(fdum)));
	scale = 1.0;

/* The line below is the difference between brcubic.c and the utility
   cubic.c where scale = pow(10.0,power) only if fabs(power) >= 9 */

	scale = pow(10.0,power);
	scalesq = scale*scale;

	a = *(coef)/scale; b = *(coef+1)/scalesq; c = *(coef+2)/scalesq/scale;

	maga = fabs(a);
	magb = fabs(b+maga*maga*SMALLER);
	magc = fabs(c+maga*magb*SMALLER);
	/*printf("a, b, c, scale, power %e %e %e %e %e\n", a, b, c, scale, power);
	printf("maga, magb, magc %e %e %e \n", maga, magb, magc);*/

	Q = (a*a - 3.0*b)/9.0;

	if( Q < SMALLEST )
	{
/* All three roots are equal */
		*(coef) = a*scale*THIRD;
		*(coef+1) = a*scale*THIRD;
		*(coef+2) = a*scale*THIRD;
		return 1;
	}

	Q3 = Q*Q*Q;
	rt2Q = sqrt(Q);
	R = (2.0*a*a*a - 9.0*a*b+27.0*c)/54.0;
	/*printf("Q,Q3,rt2Q,R %e %e %e %e \n", Q,Q3,rt2Q,R);*/

	if( Q3 + SMALL < R*R )
	{
		printf( "2 roots are complex %e %e\n",R*R, Q3);
		/*exit(1);*/
	}

	ratio = R/sqrt(Q3);
	if( fabs(ratio) > 1.0 )
	{
		ratio /= fabs(ratio);
	}
	theta = acos(ratio);
	/*printf("theta %e \n", theta);*/

	*(coef) = -2.0*rt2Q*cos(theta/3.0) - a/3;
	*(coef+1) = -2.0*rt2Q*cos((theta+2*PI)/3.0) - a/3;
	*(coef+2) = -2.0*rt2Q*cos((theta-2*PI)/3.0) - a/3;

/* Put roots in order */

	i = 0; j = 1; k = 2;
	maxroot = fabs(*(coef));
	midroot = fabs(*(coef+1));
	minroot = fabs(*(coef+2));
	if ( maxroot < fabs(*(coef+2)))
	{
		maxroot = fabs(*(coef+2));
		minroot = fabs(*(coef));
		i = 2; k = 0;
	}
	if ( maxroot < fabs(*(coef+1)))
	{
		maxroot = fabs(*(coef+1));
		midroot = fabs(*(coef+i));
		j = i; i = 1;
	}
	if ( minroot > fabs(*(coef+j)))
	{
		minroot = fabs(*(coef+j));
		midroot = fabs(*(coef+k));
		k = j; j = 2; 
	}
	fdum1 = *(coef+i);
	fdum2 = *(coef+j);
	fdum3 = *(coef+k);
	*(coef) = fdum1;
	*(coef+1) = fdum2;
	*(coef+2) = fdum3;

/* If there is a large difference in order of b and c, set smallest
   root to zero */
	if ( magc < magb*maga*SMALL )
	{
		
		*(coef+2) = 0.0;
/* If there is a large difference in order of a and b, set second smallest
   root to zero */
		if ( magb < maga*maga*SMALL )
		{
			*(coef+1) = 0.0;
		}
	}
	*(coef) *= scale;
	*(coef+1) *= scale;
	*(coef+2) *= scale;

	return 1;
}


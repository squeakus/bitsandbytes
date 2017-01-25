/* This program prints out the eigenvalue data from the Lanczos codes.

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "eigen.h"

#define pi              3.141592654

int T_print( double *T_diag, double *T_upper, int n, int num_eigen)
{
/*
                        Updated 10/6/00 
*/

	int i, j;

	printf("This is the T matrix before QR decomposition \n");
	for( i = 0; i < num_eigen; ++i )
	{
	    printf( "%3d  ", i);
	    for( j = 0; j < i-1; ++j )
	    {
		 printf( "%14.6e ",0.0);
	    }
	    if(i) printf( "%14.6e ",*(T_upper+ i-1));
	    printf( "%14.6e ",*(T_diag+i));
	    if(i < num_eigen - 1)
	    {
		printf( "%14.6e ",*(T_upper+i));
	    }
	    for( j = i+1; j < num_eigen-1; ++j )
	    {
		 printf( "%14.6e ",0.0);
	    }
	    printf( "\n");
	}
	printf( "\n");

	return 1;
}

int T_eval_print( double *T_diag, double *T_upper, int n, int num_eigen)
{
/*
                        Updated 10/6/00 
*/
	int i, j;

	printf("This is the T matrix \n");
	for( i = 0; i < num_eigen; ++i )
	{
	    printf( "%3d  ", i);
	    for( j = 0; j < i-1; ++j )
	    {
		 printf( "%14.6e ",0.0);
	    }
	    if(i) printf( "%14.6e ",*(T_upper+ i-1));
	    printf( "%14.6e ",1.0/(*(T_diag+i)));
	    if(i < num_eigen - 1)
	    {
		printf( "%14.6e ",*(T_upper+i));
	    }
	    for( j = i+1; j < num_eigen-1; ++j )
	    {
		 printf( "%14.6e ",0.0);
	    }
	    printf( "\n");
	}
	printf( "\n");

	return 1;
}

int T_evector_print(double *T_evector, int n, int num_eigen)
{
/*
                        Updated 10/6/00 
*/
	int i, j;

	printf( "\n");
	printf("These are the eigenvectors of T\n");
	for( i = 0; i < num_eigen; ++i )
	{
	    for( j = 0; j < num_eigen; ++j )
	    {
		printf( "%14.6e ",*(T_evector + num_eigen*i + j));
	    }
	    printf( "\n");
	}   
	printf( "\n");

	return 1;
}

int Lanczos_vec_print(double *q, int n, int num_eigen)
{
/*
                        Updated 10/6/00 
*/
	int i, j;

	printf( "\n");
	printf("These are the lanczos vectors\n");

/* I'm actually writing out the transpose of how I'm storing it */

	for( i = 0; i < n; ++i )
	{
	    for( j = 0; j < num_eigen; ++j )
	    {
		printf( "%14.6e ",*(q + i + n*j));
	    }
	    printf( "\n");
	}   
	printf( "\n");

	return 1;
}

int evector_print(double *ritz, int n, int num_eigen)
{
/*
  This subroutine prints the eigenvectors of the problem for both
  before and after they are re-ordered.

                        Updated 10/6/00 
*/
	int i, j;

	printf( "\n");
	for( i = 0; i < n; ++i )
	{
	    for( j = 0; j < num_eigen; ++j )
	    {
		printf( "%14.6e ",*(ritz + num_eigen*i + j));
	    }
	    printf( "\n");
	}   
	printf( "\n");

	return 1;
}

int eval_print(EIGEN *eigen, int num_eigen)
{
/*
                        Updated 10/6/00 
*/
	int i;

	printf("These are the eigenvalues in order\n");
	for( i = 0; i < num_eigen; ++i )
	{
		printf( "%14.6e ",eigen[i].val);
	}
	printf( "\n\n");
	for( i = 0; i < num_eigen; ++i )
	{
		printf( "%3d ",eigen[i].index);
	}
	printf( "\n\n");

	return 1;
}

int eval_data_print(EIGEN *eigen, char *name, int nmode)
{
/*
                        Updated 11/11/00 
*/
	int i, name_length;
	char name_eval[30], *ccheck;
	double fdum, fdum2;
	FILE *o1;

	memset(name_eval,0,30*sizeof(char));

	name_length = strlen(name);
	if( name_length > 20) name_length = 20;

	ccheck = strncpy(name_eval, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(name_eval+name_length, ".eig", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o1 = fopen( name_eval,"w" );

	fprintf(o1, "These are the eigenvalues and frequencies for the file: %s\n\n",name);
	fprintf(o1, "\n\n                        Eigenvalue");
	fprintf(o1, "          Frequency           Frequency/(2.0*pi)");
	fprintf(o1, "\n                                    ");
	fprintf(o1, "        sqrt(Eigenvalue)\n");

	for( i = 0; i < nmode; ++i )
	{
	    if(eigen[i].val > 0)
	    {
		fdum = sqrt(eigen[i].val);
		fdum2 = fdum/(2.0*pi);
		fprintf(o1, "\n mode number %4d     %16.8e    %16.8e    %16.8e",
			i+1,eigen[i].val, fdum, fdum2);
	    }
	    else
	    {
		fprintf(o1, "\n mode number %4d     %16.8e < 0     NOT VALID",
			i+1,eigen[i].val);
		fprintf(o1, "           NOT VALID");
	    }
	}
	printf( "\n");

	return 1;
}


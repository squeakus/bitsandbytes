/*
   This subroutine takes the dot product of 2 vectors 

                Updated 9/29/99

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

int dotX( double *C, double *A, double *B, int n)
{
	int i;

	*C=0.0;
	for( i = 0; i < n; ++i  ) *C += *(A+i)*(*(B+i));
	return 1;
}


/* This utility function assembles the global K matrix for
    every FEA program.

    This function hopefully should make it easier to do
    a hybrid code that consists of multiple elements.
    For this purpose, lm is not passed directly, but
    rather (lm + k*neqel), so I only have to deal
    with DOFs and not elements(k is the element number,
    renamed el here), which is required by a hybrid code.  
    The reason k is passed at all is for when the conjugate
    gradient method is used instead of LU decomp with
    skyline.

		Updated 7/6/00

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

#define MAX(x,y) (((x)>(y))?(x):(y))

int globalKassemble(double *A, int *idiag, double *K_el, int *lm, int neqel)
{
	int i, j;
	int ijmax, ijabs, locina, lmi, lmj;

/* Assembly of the global skylined stiffness matrix */

	for( i = 0; i < neqel; ++i )
	{
	    if( *(lm+i) > -1 )
	    {
		for( j = i; j < neqel; ++j )
		{
		    if( *(lm+j) > -1 )
		    {
			lmi=*(lm+i);
			lmj=*(lm+j);
			ijmax = MAX(lmi,lmj);
			ijabs = abs(lmi-lmj);
			locina = *(idiag + ijmax) - ijabs;
			*(A + locina) = *(A + locina) + *(K_el+neqel*i+j);
			/*printf("%3d %14.5f %14.5f\n ",locina,
			 + locina), *(K_el+neqel*i+j));*/
		    }
		}
	    }
	}
	return 1;
}

int globalConjKassemble(double *A, int *dof_el, int el, double *K_diag,
	double *K_el, int neqel, int neqlsq, int numel_K)
{
	int i, j;

/* Store numel_K element stiffness matrices for the Conjugate Gradient method */

/* Compute the K_diag matrix made up of the diagonal of the global K */

	for( j = 0; j < neqel; ++j )
	{
		*(K_diag + *(dof_el+j)) += *(K_el + neqel*j + j);
	}

/* Compute A to be made up of the first numel_K element stiffness K's */

	if( el < numel_K )
	{
		for( j = 0; j < neqlsq; ++j )
		{
		    *(A + el*neqlsq + j) = *(K_el + j);
		}
	}

	return 1;
}


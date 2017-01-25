/*
    This program, like rotate.c optmizes the rotation multiplication
    for a rotation matrix which has a 2X3 rather than a 3X3 non-zero
    sub matrix.

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

#define nsd     3

int matXrot2(double *C, double *A, double *rotate, int n, int neqel)
{
/*
    This subroutine multiplies A with the rotation matrix for a 3D
    triangle.  It is streamlined so that only the non-zero terms of
    the rotation matix are multiplied.

		A*[rotation] 

                A(n,neqel2) rotation(neqel2,neqel)

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 10/22/05
*/
	int i, j, p, neqel2;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);

	neqel2 = p*(ndof-1);

	memset(C, 0, n*neqel*sizeof(double));
	for( i = 0; i < n; ++i )
	{
	   for( j = 0; j < p; ++j )
	   {
		   *(C + neqel*i + nsd*j) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 3));

		   *(C + neqel*i + nsd*j + 1) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate + 1)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 4));

		   *(C + neqel*i + nsd*j + 2) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate + 2)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 5));
	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotXmat2(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix
    for a 3D triangle.  It is streamlined so that only the non-zero
    terms of the rotation matix are multiplied.

		[rotation] * A

                rotation(neqel2,neqel) A(neqel,m) 

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 10/22/05
*/

	int i, j, p, neqel2;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);
	neqel2 = p*(ndof-1);

	memset(C, 0, neqel2*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*(nsd-1)*i + j) =
			*(rotate)*(*(A + m*nsd*i + j)) + 
			*(rotate + 1)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 2)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*(nsd-1)*i + 1*m + j) =
			*(rotate + 3)*(*(A + m*nsd*i + j)) + 
			*(rotate + 4)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 5)*(*(A + m*nsd*i + 2*m + j));

	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotTXmat2(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix
    for a 3D triangle.  It is streamlined so that only the non-zero
    terms of the rotation matix are multiplied.

		[rotation(transpose)] * A

                rotation(neqel2,neqel) A(neqel2,m) 

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 8/11/06
*/

	int i, j, p, neqel2;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);
	neqel2 = p*(ndof-1);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*nsd*i + j) =
			*(rotate)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 3)*(*(A + m*(nsd-1)*i + 1*m + j));

		   *(C + m*nsd*i + 1*m + j) =
			*(rotate + 1)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 4)*(*(A + m*(nsd-1)*i + 1*m + j));

		   *(C + m*nsd*i + 2*m + j) =
			*(rotate + 2)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 5)*(*(A + m*(nsd-1)*i + 1*m + j));
	   }
	   /*printf("\n");*/
	}
	return 1;
}


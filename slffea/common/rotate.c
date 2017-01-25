/*
    This program optmizes the rotation multiplication
    for a rotation matrix which has a 3X3 non-zero sub matrix.  It
    is used primarily in the 3D beam and truss codes.

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

#define nsd     3

int matXrot(double *C, double *A, double *rotate, int n, int neqel)
{
/*
    This subroutine multiplies A with the rotation matrix for a 3D
    beam and truss.  It is streamlined so that only the non-zero terms of
    the rotation matix are multiplied.

		A*[rotation] 

                A(n,neqel) rotation(neqel,neqel)

    The rotate matrix below is a 3X3 matrix.  "p" of these 3X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 9/28/00
*/
	int i, j, p;

	p = (int)((double)neqel/3.0);

	memset(C, 0, n*neqel*sizeof(double));
	for( i = 0; i < n; ++i )
	{
	   for( j = 0; j < p; ++j )
	   {
		   *(C + neqel*i + nsd*j) =
			*(A + neqel*i + nsd*j)*(*(rotate)) + 
			*(A + neqel*i + nsd*j + 1)*(*(rotate + 3)) + 
			*(A + neqel*i + nsd*j + 2)*(*(rotate + 6));

		   *(C + neqel*i + nsd*j + 1) =
			*(A + neqel*i + nsd*j)*(*(rotate + 1)) + 
			*(A + neqel*i + nsd*j + 1)*(*(rotate + 4)) + 
			*(A + neqel*i + nsd*j + 2)*(*(rotate + 7));

		   *(C + neqel*i + nsd*j + 2) =
			*(A + neqel*i + nsd*j)*(*(rotate + 2)) + 
			*(A + neqel*i + nsd*j + 1)*(*(rotate + 5)) + 
			*(A + neqel*i + nsd*j + 2)*(*(rotate + 8));
	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotXmat(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix
    for a 3D beam and truss.  It is streamlined so that only the
    non-zero terms of the rotation matix are multiplied.

		[rotation] * A

               rotation(neqel,neqel) A(neqel,m) 

    The rotate matrix below is a 3X3 matrix.  "p" of these 3X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 9/28/00
*/

	int i, j, p;

	p = (int)((double)neqel/3.0);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*nsd*i + j) =
			*(rotate)*(*(A + m*nsd*i + j)) + 
			*(rotate + 1)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 2)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*nsd*i + 1*m + j) =
			*(rotate + 3)*(*(A + m*nsd*i + j)) + 
			*(rotate + 4)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 5)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*nsd*i + 2*m + j) =
			*(rotate + 6)*(*(A + m*nsd*i + j)) + 
			*(rotate + 7)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 8)*(*(A + m*nsd*i + 2*m + j));
	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotTXmat(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix for
    a 3D beam and truss.  It is streamlined so that only the non-zero
    terms of the rotation matix are multiplied.

		[rotation(transpose)] * A

		rotation(transpose)(neqel,neqel) A(neqel,m) 

		neqel = npel*ndof

		where ndof = 3

    The rotate matrix below is a 3X3 matrix.  "p" of these 3X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 8/11/06
*/

	int i, j, p;

	p = (int)((double)neqel/3.0);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*nsd*i + j) =
			*(rotate)*(*(A + m*nsd*i + j)) + 
			*(rotate + 3)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 6)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*nsd*i + 1*m + j) =
			*(rotate + 1)*(*(A + m*nsd*i + j)) + 
			*(rotate + 4)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 7)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*nsd*i + 2*m + j) =
			*(rotate + 2)*(*(A + m*nsd*i + j)) + 
			*(rotate + 5)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 8)*(*(A + m*nsd*i + 2*m + j));
	   }
	   /*printf("\n");*/
	}
	return 1;
}


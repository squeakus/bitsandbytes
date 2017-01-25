/*
    This program, like rotate.c optmizes the rotation multiplication
    for a rotation matrix which has a 3X3 non-zero sub matrix.  It
    is somewhat hardcoded for 3-D plate elements which have unique
    rotation needs in that the 1st 3 degrees of freedom of a node
    corresponding to displacement are rotated, but not the last 2
    which are for rotations which remain local to the coordinates
    of the plane of the plate.

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

#define nsd        3
#define ndof5      5
#define ndof3      3
#define neqel12   12

int matXrot4(double *C, double *A, double *rotate, int n, int neqel)
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

	Updated 9/11/06
*/
	int i, j, k, p;

	p = (int)((double)neqel/5.0);

	memset(C, 0, n*neqel*sizeof(double));
	for( i = 0; i < n; ++i )
	{
		for( j = 0; j < p; ++j )
		{
		   *(C + neqel*i + ndof5*j) =
			*(A + neqel12*i + ndof3*j)*(*(rotate + 6));

		   *(C + neqel*i + ndof5*j + 1) =
			*(A + neqel12*i + ndof3*j)*(*(rotate + 7));

		   *(C + neqel*i + ndof5*j + 2) =
			*(A + neqel12*i + ndof3*j)*(*(rotate + 8));

		   *(C + neqel*i + ndof5*j + 3) =
			*(A + neqel12*i + ndof3*j + 1);

		   *(C + neqel*i + ndof5*j + 4) =
			*(A + neqel12*i + ndof3*j + 2);
		}
		/*printf("\n");*/
	}
	return 1;
}

int rotXmat4(double *C, double *rotate, double *A, int m, int neqel)
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

	Updated 9/8/06
*/

	int i, j, k, p;

	p = (int)((double)neqel/5.0);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	    for( j = 0; j < m; ++j )
	    {
		   *(C + m*ndof3*i + j) =
			*(rotate + 6)*(*(A + m*ndof5*i + j)) + 
			*(rotate + 7)*(*(A + m*ndof5*i + 1*m + j)) + 
			*(rotate + 8)*(*(A + m*ndof5*i + 2*m + j));

		   *(C + m*ndof3*i + 1*m + j) = *(A + m*ndof5*i + 3*m + j);

		   *(C + m*ndof3*i + 2*m + j) = *(A + m*ndof5*i + 4*m + j);
	    }
		/*printf("\n");*/
	}
	return 1;
}

#if 1

int rotTXmat4(double *C, double *rotate, double *A, int m, int neqel)
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

	Updated 9/8/06
*/

	int i, j, p;

	p = (int)((double)neqel/5.0);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*ndof5*i + j) =
			*(rotate + 6)*(*(A + m*ndof3*i + j)); 

		   *(C + m*ndof5*i + 1*m + j) =
			*(rotate + 7)*(*(A + m*ndof3*i + j)); 

		   *(C + m*ndof5*i + 2*m + j) =
			*(rotate + 8)*(*(A + m*ndof3*i + j)); 

		   *(C + m*ndof5*i + 3*m + j) = *(A + m*ndof3*i + 1*m + j);

		   *(C + m*ndof5*i + 4*m + j) = *(A + m*ndof3*i + 2*m + j);
	   }
	   /*printf("\n");*/
	}
	return 1;
}

#endif

#if 0


int rotTXmat4(double *C, double *rotate, double *A, int m, int neqel)
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

	Updated 9/8/06
*/

	int i, j, k, p, p2;

	p = (int)((double)neqel/5.0);
	p2 = (int)((double)m/5.0);

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < p2; ++j )
	   {
		for( k = 0; k < nsd; ++k )
		{
		   *(C + m*ndof5*i + ndof5*j + k) =
			*(rotate)*(*(A + m*ndof5*i + ndof5*j + k)) + 
			*(rotate + 3)*(*(A + m*ndof5*i + 1*m + ndof5*j + k)) + 
			*(rotate + 6)*(*(A + m*ndof5*i + 2*m + ndof5*j + k));

		   *(C + m*ndof5*i + 1*m + ndof5*j + k) =
			*(rotate + 1)*(*(A + m*ndof5*i + ndof5*j + k)) + 
			*(rotate + 4)*(*(A + m*ndof5*i + 1*m + ndof5*j + k)) + 
			*(rotate + 7)*(*(A + m*ndof5*i + 2*m + ndof5*j + k));

		   *(C + m*ndof5*i + 2*m + ndof5*j + k) =
			*(rotate + 2)*(*(A + m*ndof5*i + ndof5*j + k)) + 
			*(rotate + 5)*(*(A + m*ndof5*i + 1*m + ndof5*j + k)) + 
			*(rotate + 8)*(*(A + m*ndof5*i + 2*m + ndof5*j + k));

		   *(C + m*ndof5*i + 3*m + ndof5*j + k) =
			*(A + m*ndof5*i + 3*m + ndof5*j + k);

		   *(C + m*ndof5*i + 4*m + ndof5*j + k) =
			*(A + m*ndof5*i + 4*m + ndof5*j + k);
		}
		for( k = nsd; k < ndof5; ++k )
		{
		   *(C + m*ndof5*i + ndof5*j + k) =
			*(A + m*ndof5*i + ndof5*j + k);

		   *(C + m*ndof5*i + 1*m + ndof5*j + k) =
			*(A + m*ndof5*i + 1*m + ndof5*j + k);

		   *(C + m*ndof5*i + 2*m + ndof5*j + k) =
			*(A + m*ndof5*i + 2*m + ndof5*j + k);

		   *(C + m*ndof5*i + 3*m + ndof5*j + k) =
			*(A + m*ndof5*i + 3*m + ndof5*j + k);

		   *(C + m*ndof5*i + 4*m + ndof5*j + k) =
			*(A + m*ndof5*i + 4*m + ndof5*j + k);
		}
	   }
	   /*printf("\n");*/
	}
	return 1;
}

#endif

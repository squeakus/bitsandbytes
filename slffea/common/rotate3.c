/*
    This program, like rotate.c optmizes the rotation multiplication
    for a rotation matrix which has a 2X3 rather than a 3X3 non-zero
    sub matrix.  It is used primarily in the quadrilateral code.  Another
    reason it is different from rotate.c and rotate2.c is that the
    non-zero submatrix is different for each node because each node has
    its own local coordinate system.  The 1st row of the rotation matrix is
    made up of the local x basis and the 2nd row is the y basis.  One thing
    that should be mentioned is that all these basis vectors should be
    lying in the same plane of the quad element because there is no out of
    plane bending. 

    I am no longer using this code.  ANSYS provides a better method for
    having a local basis which is the same for all the nodes.  I elaborate
    more on this in:

       ~/slffea-1.4/quad/quad/qdkasmbl.c

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
#define nsd2    2

int matXrot3(double *C, double *A, double *rotate, int n, int neqel)
{
/*
    This subroutine multiplies A with the rotation matrix for a 3D
    quadrilateral.  It is streamlined so that only the non-zero terms of
    the rotation matix are multiplied.

		A*[rotation] 

                A(n,neqel2) rotation(neqel2,neqel)

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 8/15/06
*/
	int i, j, p, neqel2, dum;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);
	dum = nsd2*nsd;

	neqel2 = p*(ndof-1);

	memset(C, 0, n*neqel*sizeof(double));
	for( i = 0; i < n; ++i )
	{
	   for( j = 0; j < p; ++j )
	   {
		   *(C + neqel*i + nsd*j) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate + j*dum)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 3 + j*dum));

		   *(C + neqel*i + nsd*j + 1) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate + 1 + j*dum)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 4 + j*dum));

		   *(C + neqel*i + nsd*j + 2) =
			*(A + neqel2*i + (nsd-1)*j)*(*(rotate + 2 + j*dum)) + 
			*(A + neqel2*i + (nsd-1)*j + 1)*(*(rotate + 5 + j*dum));

		   /*printf("%4d %4d %d\n", i, j, j*dum);
		   printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
			*(rotate + j*dum), *(rotate + 3 + j*dum), *(rotate + 1 + j*dum),
			*(rotate + 4 + j*dum), *(rotate + 2 + j*dum), *(rotate + 5 + j*dum));
		   printf("%14.6e %14.6e\n",
			*(A + neqel2*i + (nsd-1)*j), *(A + neqel2*i + (nsd-1)*j + 1));
		   printf("%4d %4d %4d\n", neqel*i + nsd*j, neqel*i + nsd*j + 1, neqel*i + nsd*j + 2);
		   printf("%14.6e %14.6e %14.6e\n",
			*(C + neqel*i + nsd*j), *(C + neqel*i + nsd*j + 1), *(C + neqel*i + nsd*j + 2));*/
	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotXmat3(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix
    for a 3D quadrilateral.  It is streamlined so that only the non-zero
    terms of the rotation matix are multiplied.

		[rotation] * A

                rotation(neqel2,neqel) A(neqel,m) 

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 8/15/06
*/

	int i, j, p, neqel2, dum;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);
	neqel2 = p*(ndof-1);
	dum = nsd2*nsd;

	memset(C, 0, neqel2*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*(nsd-1)*i + j) =
			*(rotate + i*dum)*(*(A + m*nsd*i + j)) + 
			*(rotate + 1 + i*dum)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 2 + i*dum)*(*(A + m*nsd*i + 2*m + j));

		   *(C + m*(nsd-1)*i + 1*m + j) =
			*(rotate + 3 + i*dum)*(*(A + m*nsd*i + j)) + 
			*(rotate + 4 + i*dum)*(*(A + m*nsd*i + 1*m + j)) + 
			*(rotate + 5 + i*dum)*(*(A + m*nsd*i + 2*m + j));

	   }
	   /*printf("\n");*/
	}
	return 1;
}

int rotTXmat3(double *C, double *rotate, double *A, int m, int neqel)
{
/*
    This subroutine multiplies the rotation matrix with the A matrix
    for a 3D quadrilateral.  It is streamlined so that only the non-zero
    terms of the rotation matix are multiplied.

		[rotation(transpose)] * A

                rotation(neqel2,neqel) A(neqel2,m) 

                neqel = npel*ndof
                neqel2 = npel*(ndof-1)

                where usually, ndof = 3

    The rotate matrix below is a 2X3 matrix.  "p" of these 2X3
    matrices lie on the diagonal of the above global "rotation"
    matrix.

	Updated 8/15/06
*/

	int i, j, p, neqel2, dum;
	int ndof = nsd;

	p = (int)((double)neqel/3.0);
	neqel2 = p*(ndof-1);
	dum = nsd2*nsd;

	memset(C, 0, neqel*m*sizeof(double));
	for( i = 0; i < p; ++i )
	{
	   for( j = 0; j < m; ++j )
	   {
		   *(C + m*nsd*i + j) =
			*(rotate + i*dum)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 3 + i*dum)*(*(A + m*(nsd-1)*i + 1*m + j));

		   *(C + m*nsd*i + 1*m + j) =
			*(rotate + 1 + i*dum)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 4 + i*dum)*(*(A + m*(nsd-1)*i + 1*m + j));

		   *(C + m*nsd*i + 2*m + j) =
			*(rotate + 2 + i*dum)*(*(A + m*(nsd-1)*i + j)) + 
			*(rotate + 5 + i*dum)*(*(A + m*(nsd-1)*i + 1*m + j));

		   /*printf("%4d %4d\n", i, j);
		   printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
			*(rotate + i*dum), *(rotate + 3 + i*dum), *(rotate + 1 + i*dum),
			*(rotate + 4 + i*dum), *(rotate + 2 + i*dum), *(rotate + 5 + i*dum));
		   printf("%14.6e %14.6e\n",
			*(A + m*(nsd-1)*i + j), *(A + m*(nsd-1)*i + 1*m + j));
		   printf("%4d %4d %4d\n", m*nsd*i + j, m*nsd*i + 1*m + j, m*nsd*i + 2*m + j);
		   printf("%14.6e %14.6e %14.6e\n",
			*(C + m*nsd*i + j), *(C + m*nsd*i + 1*m + j), *(C + m*nsd*i + 2*m + j));*/
	   }
	   /*printf("\n");*/
	}
	return 1;
}


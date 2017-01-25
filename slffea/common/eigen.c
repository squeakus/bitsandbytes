/* This program compares two inputs x and y for the qsort call
   in the *lanczos subroutines.  This is ultimately used to
   sort the eigenvalues calculated from modal analysis in
   ascending order.

                        Updated 8/16/00

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/


#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "eigen.h"

int compare_eigen ( const void *x,  const void *y)
{
	EIGEN *a = (EIGEN *) x;
	EIGEN *b = (EIGEN *) y;
	if ( a[0].val < b[0].val ) return -1;
	if ( a[0].val == b[0].val ) return 0;
	if ( a[0].val > b[0].val ) return 1;
}

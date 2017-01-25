/*
    This file contains the structures of the eigenvalue comparison
    subroutine.  This is ultimately used to sort the eigenvalues
    calculated from modal analysis in ascending order.

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

typedef struct {
	double val;
	int index;
} EIGEN;


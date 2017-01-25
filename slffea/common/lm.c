/*
    This utility function assembles the lm array for all finite 
    element programs

		Updated 12/15/98

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

int formlm( int *connect, int *id, int *lm, int ndof, int npel, int numel )
{
/* Assembly of the lm array(the matrix which gives 
   the degree of feedom per element and node ) */

	int i,j,k,node;

	for( k = 0; k < numel; ++k )
	{
	   for( j = 0; j < npel; ++j )
	   {
		node = *(connect + npel*k + j);
		for( i = 0; i < ndof; ++i )
		{
			*(lm + ndof*npel*k + ndof*j + i) = *(id + ndof*node + i );
		}
	   }
	}
	return 1;
}

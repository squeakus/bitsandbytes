/* This program reapplies the boundary conditions for the
   displacement when the conjugate gradient method is used.
   This is for a finite element program which does analysis
   on a tetrahedral element.

                        Updated 3/19/01 

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
#include "teconst.h"
#if TETRA1
#include "testruct.h"
#endif
#if TETRA2
#include "../tetra2/te2struct.h"
#endif

int teBoundary ( double *U, BOUND bc )
{
	int i,dum;

        for( i = 0; i < bc.num_fix[0].x; ++i )
        {
            dum = bc.fix[i].x;
            *(U+ndof*dum) = 0.0;
        }
        for( i = 0; i < bc.num_fix[0].y; ++i )
        {
            dum = bc.fix[i].y;
            *(U+ndof*dum+1) = 0.0;
        }
        for( i = 0; i < bc.num_fix[0].z; ++i )
        {
            dum = bc.fix[i].z;
            *(U+ndof*dum+2) = 0.0;
        }
	return 1;
}

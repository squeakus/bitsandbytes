/* This program reapplies the boundary conditions for the
   temperature when the conjugate gradient method is used.
   This is for a finite element program which does analysis
   on the thermal brick element.

	        Updated 2/18/00 

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../brick/brconst.h"
#include "br2struct.h"

int br2Boundary ( double *T, BOUND bc )
{

	int i,dum;

	for( i = 0; i < bc.num_T[0]; ++i )
	{
	    dum = bc.T[i];
	    *(T+Tndof*dum) = 0.0;
	}

	return 1;
}


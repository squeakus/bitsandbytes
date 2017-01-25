/* This program reapplies the boundary conditions for the
   displacement when the conjugate gradient method is used.
   This is for a finite element program which does analysis
   on an electromagnetic quad element.

                        Updated 12/7/00 

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qd3const.h"
#include "qd3struct.h"

int qd3Boundary ( double *EMedge, BOUND bc )
{
	int i,dum;

        for( i = 0; i < bc.num_edge[0]; ++i )
        {
            dum = bc.edge[i];
            *(EMedge + edof*dum) = 0.0;
        }
	return 1;
}

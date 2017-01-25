/*
    This utility function assembles the id array for a finite 
    element program which does analysis on a 8 node thermal
    brick element.

		Updated 6/20/02

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

extern int Tdof, Tneqn;

int brformTid( BOUND bc, int *id)
{
/* Assembly of the id array(the matrix which determines
   the degree of feedom by setting fixed nodes = -1) */

	int i, counter;

        counter=0;
        for( i = 0; i < bc.num_T[0]; ++i )
	{
           *(id + Tndof*bc.T[i]) = -1;
	} 
        for( i = 0; i < Tdof; ++i )
	{
           if( *(id + i) != -1  )
           {
                *(id + i) = counter;
                ++counter;
           }
	}
        Tneqn=counter;
	return 1;
}


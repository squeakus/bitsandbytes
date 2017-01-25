/*
    This utility function assembles the id and lm arrays for a finite 
    element program which does analysis on a 2 node truss element. 

		Updated 12/15/98

    SLFFEA source file
    Version:  1.0
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tsconst.h"
#include "tsstruct.h"

extern int dof, neqn, numel, numnp;

int tsformid( BOUND bc, int *id )
{
/* Assembly of the id array(the matrix which determines
   the degree of feedom by setting fixed nodes = -1) */

	int i, counter;

        counter=0;
        for( i = 0; i < bc.num_fix[0].x; ++i )
	{
           *(id + ndof*bc.fix[i].x) = -1;
	} 
        for( i = 0; i < bc.num_fix[0].y; ++i )
	{
           *(id + ndof*bc.fix[i].y + 1) = -1;
	} 
        for( i = 0; i < bc.num_fix[0].z; ++i )
	{
           *(id + ndof*bc.fix[i].z + 2) = -1;
	} 
        for( i = 0; i < dof; ++i )
	{
           if( *(id + i) != -1  )
           {
                *(id + i) = counter;
                ++counter;
           }
	}
        neqn=counter;
	return 1;
}

int formlm( int *connect, int *id, int *lm )

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

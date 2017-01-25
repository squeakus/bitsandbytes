/*
   This utility function assembles the idiag array for a finite 
   element program which does analysis on a 2 node truss
   element 

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

extern int neqn, numel, numnp;

int diag( int *idiag, int *lm )

{
	int i,j,k,m,node,min,num;

/* Assembly of the idiag array for the skyline */

        for( i = 0; i < neqn; ++i )
	{
		*(idiag+i)=i;	
	}
        for( k = 0; k < numel; ++k )
	{
	   min=neqn;
           for( j = 0; j < npel; ++j )
	   {
           	for( i = 0; i < ndof; ++i )
	   	{
			num=*(lm + ndof*npel*k + ndof*j + i);
			if(num > -1 )
			{
				min = MIN(min,num);
			}
	   	}
	   }
           for( j = 0; j < npel; ++j )
	   {
           	for( i = 0; i < ndof; ++i )
	   	{
			num=*(lm + ndof*npel*k + ndof*j + i);
			if(num > -1 )
                        {
                                *(idiag+num) = MIN(*(idiag+num),min);
                        }
	   	}
	   }
	}
        for( i = 1; i < neqn; ++i )
	{
		*(idiag+i)=*(idiag+i-1)+i-*(idiag+i)+1;	
	}
	return 1;
}


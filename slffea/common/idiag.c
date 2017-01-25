/*
    This utility function assembles the idiag array for all finite 
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

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

int diag( int *idiag, int *lm, int ndof, int neqn, int npel, int numel)
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


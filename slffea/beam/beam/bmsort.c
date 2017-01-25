/* This utility function sorts a set of integers.
   It reads in the data and writes out the ordered set.

                Written by San Le
                slffea@juno.com
                www.geocities.com/Athens/2099

    Copyright (C) 1999, 2000  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

                Updated 4/22/00
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define big       1e7

int bmsort(int *numbers, int n)
{
        int i, range, k, maximum, minimum, shift, *temp, dum;

	maximum = -big;
        minimum = big;

        for( i = 0; i < n; ++i )
        {
		if( maximum < *(numbers+i))
		{
		 	maximum = *(numbers+i);
		}
		if( minimum > *(numbers+i))
		{
		 	minimum = *(numbers+i);
		}
        }
	range = maximum - minimum + 1;
	temp=(int *)calloc(range,sizeof(int));
	shift = - minimum;
        for( i = 0; i < n; ++i )
        {
		*(temp + *(numbers+i) + shift) = -1;
	}
	*(numbers ) = minimum;
	*(numbers + n - 1) = maximum;
	dum = 1;
        for( i = 1; i < range - 1; ++i )
        {
		if( *(temp + i) < 0 )
		{
		 	*(numbers + dum) = i - shift;
			++dum;
		}
	}
	free(temp);
	return 1;
}


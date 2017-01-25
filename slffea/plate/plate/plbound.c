/* This program reapplies the boundary conditions for the
   displacement when the conjugate gradient method is used.
   This is for a finite element program which does analysis
   on a plate element.

                        Updated 11/1/06 

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "plconst.h"
#include "plstruct.h"

int plBoundary ( double *U, BOUND bc )
{
	int i,dum;

	for( i = 0; i < bc.num_fix[0].x; ++i )
	{
	    dum = bc.fix[i].x;
	    *(U+ndof6*dum) = 0.0;
	}
	for( i = 0; i < bc.num_fix[0].y; ++i )
	{
	    dum = bc.fix[i].y;
	    *(U+ndof6*dum+1) = 0.0;
	}
	for( i = 0; i < bc.num_fix[0].z; ++i )
	{
	    dum = bc.fix[i].z;
	    *(U+ndof6*dum+2) = 0.0;
	}
	for( i = 0; i < bc.num_fix[0].phix; ++i )
	{
	    dum = bc.fix[i].phix;
	    *(U+ndof6*dum+3) = 0.0;
	}
	for( i = 0; i < bc.num_fix[0].phiy; ++i )
	{
	    dum = bc.fix[i].phiy;
	    *(U+ndof6*dum+4) = 0.0;
	}
	for( i = 0; i < bc.num_fix[0].phiz; ++i )
	{
	    dum = bc.fix[i].phiz;
	    *(U+ndof6*dum+5) = 0.0;
	}
	return 1;
}

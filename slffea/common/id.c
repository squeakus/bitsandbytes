/*
    This utility function assembles the id array for 3-D
    elements that have no rotational degrees of freedom
    for the finite element program.

	        Updated 8/6/06

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

#if BRICK1
#include "../brick/brick/brconst.h"
#include "../brick/brick/brstruct.h"
#endif
#if BRICK2
#include "../brick/brick/brconst.h"
#include "../brick/brick2/br2struct.h"
#endif
#if QUAD1
#include "../quad/quad/qdconst.h"
#include "../quad/quad/qdstruct.h"
#endif
#if QUAD2
#include "../quad/quad/qdconst.h"
#include "../quad/quad2/qd2struct.h"
#endif
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#endif
#if TETRA2
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra2/te2struct.h"
#endif
#if TRI1
#include "../tri/tri/trconst.h"
#include "../tri/tri/trstruct.h"
#endif
#if TRI2
#include "../tri/tri/trconst.h"
#include "../tri/tri2/tr2struct.h"
#endif
#if TRUSS1
#include "../truss/truss/tsconst.h"
#include "../truss/truss/tsstruct.h"
#endif
#if TRUSS2
#include "../truss/truss/tsconst.h"
#include "../truss/truss2/ts2struct.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#endif
#if WEDGE2
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge2/we2struct.h"
#endif


extern int dof, neqn;

int formid( BOUND bc, int *id)
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


/*
    This utility function allocates the memory for the graphics variables for
    3-D non-constant strain elements(an example of a constant strain element is the
    tetrahedron). 

        Updated 8/12/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stdlib.h>
#include <string.h>
#if BRICK1
#include "../brick/brick/brconst.h"
#include "../brick/brick/brstruct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if BRICK2
#include "../brick/brick/brconst.h"
#include "../brick/brick2/br2struct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if QUAD1
#include "../quad/quad/qdconst.h"
#include "../quad/quad/qdstruct.h"
#include "../quad/quad_gr/qdstrcgr.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#include "../wedge/wedge_gr/westrcgr.h"
#endif

extern int input_flag, post_flag;

int Memory_gr( ISTRAIN **strain_color, ISTRESS **stress_color, int sofmISTRESS,
	NORM **mem_NORM, int sofmNORM )
{
/* Allocate the memory for when first running the program

	Updated 2/8/06

*/

/* For the ISTRESS integers */

	*stress_color=(ISTRESS *)calloc(sofmISTRESS,sizeof(ISTRESS));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}


/* For the ISTRAIN integers */

	*strain_color=(ISTRAIN *)calloc(sofmISTRESS,sizeof(ISTRAIN));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}

/* For the NORM doubles */

	*mem_NORM=(NORM *)calloc(sofmNORM, sizeof(NORM));
	if(!mem_NORM )
	{
		printf( "failed to allocate memory for NORM doubles\n ");
		exit(1);
	}

	return 1; 
}

int Memory2_gr( XYZF **mem_XYZF, int sofmXYZF )
{
	*mem_XYZF=(XYZF *)calloc(sofmXYZF,sizeof(XYZF));
	if(!mem_XYZF )
	{
		printf( "failed to allocate memory for XYZF doubles\n ");
		exit(1);
	}
	return 1; 
}


int ReGetMemory_gr( ISTRAIN **strain_color, ISTRESS **stress_color, int sofmISTRESS,
	NORM **mem_NORM, int sofmNORM )
{
/* Re-Allocate the memory for new input file 

	Updated 2/8/06

*/

/* For the ISTRESS integers */

	*stress_color=(ISTRESS *)realloc(*stress_color, sofmISTRESS*sizeof(ISTRESS));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}
	memset(*stress_color,0,sofmISTRESS*sizeof(ISTRESS));

/* For the ISTRAIN integers */

	*strain_color=(ISTRAIN *)realloc(*strain_color, sofmISTRESS*sizeof(ISTRAIN));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}
	memset(*strain_color,0,sofmISTRESS*sizeof(ISTRAIN));

/* For the NORM doubles */

	*mem_NORM=(NORM *)realloc(*mem_NORM, sofmNORM*sizeof(NORM));
	if(!mem_NORM )
	{
		printf( "failed to allocate memory for NORM doubles\n ");
		exit(1);
	}
	memset(*mem_NORM,0,sofmNORM*sizeof(NORM));

	return 1; 
}

int ReGetMemory2_gr( XYZF **mem_XYZF, int sofmXYZF )
{
	*mem_XYZF=(XYZF *)realloc(*mem_XYZF, sofmXYZF*sizeof(XYZF));
	if(!mem_XYZF )
	{
		printf( "failed to allocate memory for XYZF doubles\n ");
		exit(1);
	}
	memset(*mem_XYZF,0,sofmXYZF*sizeof(XYZF));

	return 1; 
}

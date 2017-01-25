/*
    This utility function allocates the memory for the graphics variables for
    the truss finite element graphics program.

        Updated 1/22/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tsstrcgr.h"
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"

extern int input_flag, post_flag;

int tsMemory_gr( ISDIM **strain_color, ISDIM **stress_color, int sofmISDIM )
{
/* Allocate the memory for when first running the program

        Updated 1/22/06

*/

/* For the ISDIM integers */

	*stress_color=(ISDIM *)calloc(sofmISDIM,sizeof(ISDIM));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}


/* For the ISDIM integers */

	*strain_color=(ISDIM *)calloc(sofmISDIM,sizeof(ISDIM));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}

	return 1; 
}

int tsMemory2_gr( XYZF **mem_XYZF, int sofmXYZF )
{
	*mem_XYZF=(XYZF *)calloc(sofmXYZF,sizeof(XYZF));
	if(!mem_XYZF )
	{
		printf( "failed to allocate memory for XYZF doubles\n ");
		exit(1);
	}
	return 1; 
}

int tsReGetMemory_gr( ISDIM **strain_color, ISDIM **stress_color, int sofmISDIM )
{
/* Allocate the memory for when first running the program

        Updated 1/22/06

*/

/* For the ISDIM integers */

	*stress_color=(ISDIM *)realloc(*stress_color, sofmISDIM*sizeof(ISDIM));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}
	memset(*stress_color,0,sofmISDIM*sizeof(ISDIM));

/* For the ISDIM integers */

	*strain_color=(ISDIM *)realloc(*strain_color, sofmISDIM*sizeof(ISDIM));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}
	memset(*strain_color,0,sofmISDIM*sizeof(ISDIM));

	return 1; 
}

int tsReGetMemory2_gr( XYZF **mem_XYZF, int sofmXYZF )
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

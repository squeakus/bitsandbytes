/*
    This utility function allocates the memory for the graphics variables for
    the tetrahedral finite element graphics program. 

                  Last Update 2/8/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stdlib.h>
#include <string.h>
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#include "../tetra/tetra_gr/testrcgr.h"
#endif
#if TETRA2
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/te2struct.h"
#include "../tetra/tetra_gr/testrcgr.h"
#endif
#if TRI1
#include "../tri/tri/trconst.h"
#include "../tri/tri/trstruct.h"
#include "../tri/tri_gr/trstrcgr.h"
#endif
#if TRI2
#include "../tri/tri/trconst.h"
#include "../tri/tri2/tr2struct.h"
#include "../tri/tri_gr/trstrcgr.h"
#endif



extern int input_flag, post_flag;

int Memory_gr2( ISDIM **strain_color, ISDIM **stress_color, int sofmISDIM,
	NORM **mem_NORM, int sofmNORM )
{
/* Allocate the memory for when first running the program

                  Last Update 2/8/06

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

/* For the NORM doubles */

	*mem_NORM=(NORM *)calloc(sofmNORM, sizeof(NORM));
	if(!mem_NORM )
	{
		printf( "failed to allocate memory for NORM doubles\n ");
		exit(1);
	}

	return 1; 
}

int Memory2_gr2( XYZF **mem_XYZF, int sofmXYZF )
{
	*mem_XYZF=(XYZF *)calloc(sofmXYZF,sizeof(XYZF));
	if(!mem_XYZF )
	{
		printf( "failed to allocate memory for XYZF doubles\n ");
		exit(1);
	}
	return 1; 
}


int ReGetMemory_gr2( ISDIM **strain_color, ISDIM **stress_color, int sofmISDIM,
	NORM **mem_NORM, int sofmNORM )
{
/* Re-Allocate the memory for new input file

                  Last Update 2/8/06

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

int ReGetMemory2_gr2( XYZF **mem_XYZF, int sofmXYZF )
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

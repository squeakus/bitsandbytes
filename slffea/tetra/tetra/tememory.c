/*
    This utility function allocates the memory for
    the tetrahedral finite element program.

        Updated 8/17/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "teconst.h"
#if TETRA1
#include "testruct.h"
#endif
#if TETRA2
#include "../tetra2/te2struct.h"
#endif

int teMemory( double **mem_double, int sofmf, int **mem_int, int sofmi,
	MATL **matl, int nmat, XYZI **mem_XYZI, int sofmXYZI, SDIM **strain,
	SDIM **strain_node, SDIM **stress, SDIM **stress_node,
	int sofmSDIM, int sofmSDIM_node )
{
/* For the doubles */
	*mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
	    printf( "failed to allocate memory for doubles\n ");
	    exit(1);
	}

/* For the materials */
	*matl=(MATL *)calloc(nmat,sizeof(MATL));
	if(!matl )
	{
		printf( "failed to allocate memory for matl doubles\n ");
		exit(1);
	}

/* For the SDIM doubles */

	*stress=(SDIM *)calloc(sofmSDIM,sizeof(SDIM));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}

/* For the SDIM node doubles */

	*stress_node=(SDIM *)calloc(sofmSDIM_node,sizeof(SDIM));
	if(!stress_node )
	{
		printf( "failed to allocate memory for stress_node doubles\n ");
		exit(1);
	}

/* For the SDIM doubles */

	*strain=(SDIM *)calloc(sofmSDIM,sizeof(SDIM));
	if(!strain )
	{
		printf( "failed to allocate memory for strain doubles\n ");
		exit(1);
	}

/* For the SDIM node doubles */

	*strain_node=(SDIM *)calloc(sofmSDIM_node,sizeof(SDIM));
	if(!strain_node )
	{
		printf( "failed to allocate memory for strain_node doubles\n ");
		exit(1);
	}

/* For the integers */
	*mem_int=(int *)calloc(sofmi,sizeof(int));
	if(!mem_int )
	{
		printf( "failed to allocate memory for integers\n ");
		exit(1);
	}

/* For the XYZI integers */

	*mem_XYZI=(XYZI *)calloc(sofmXYZI,sizeof(XYZI));
	if(!mem_XYZI )
	{
		printf( "failed to allocate memory for XYZI integers\n ");
		exit(1);
	}

	return 1;
}


int teReGetMemory( double **mem_double, int sofmf, int **mem_int, int sofmi,
	MATL **matl, int nmat, XYZI **mem_XYZI, int sofmXYZI, SDIM **strain,
	SDIM **strain_node, SDIM **stress, SDIM **stress_node,
	int sofmSDIM, int sofmSDIM_node )
{
/* For the doubles */
	*mem_double=(double *)realloc(*mem_double, sofmf*sizeof(double));

	if(!mem_double )
	{
	    printf( "failed to allocate memory for doubles\n ");
	    exit(1);
	}
	memset(*mem_double,0,sofmf*sizeof(double));

/* For the materials */
	*matl=(MATL *)realloc(*matl, nmat*sizeof(MATL));
	if(!matl )
	{
		printf( "failed to allocate memory for matl doubles\n ");
		exit(1);
	}
	memset(*matl,0,nmat*sizeof(MATL));

/* For the SDIM doubles */

	*stress=(SDIM *)realloc(*stress, sofmSDIM*sizeof(SDIM));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}
	memset(*stress,0,sofmSDIM*sizeof(SDIM));

/* For the SDIM node doubles */

	*stress_node=(SDIM *)realloc(*stress_node, sofmSDIM_node*sizeof(SDIM));
	if(!stress_node )
	{
		printf( "failed to allocate memory for stress_node doubles\n ");
		exit(1);
	}
	memset(*stress_node,0,sofmSDIM_node*sizeof(SDIM));

/* For the SDIM doubles */

	*strain=(SDIM *)realloc(*strain, sofmSDIM*sizeof(SDIM));
	if(!strain )
	{
		printf( "failed to allocate memory for strain doubles\n ");
		exit(1);
	}
	memset(*strain,0,sofmSDIM*sizeof(SDIM));

/* For the SDIM node doubles */

	*strain_node=(SDIM *)realloc(*strain_node, sofmSDIM_node*sizeof(SDIM));
	if(!strain_node )
	{
		printf( "failed to allocate memory for strain_node doubles\n ");
		exit(1);
	}
	memset(*strain_node,0,sofmSDIM_node*sizeof(SDIM));

/* For the integers */
	*mem_int=(int *)realloc(*mem_int, sofmi*sizeof(int));
	if(!mem_int )
	{
		printf( "failed to allocate memory for integers\n ");
		exit(1);
	}
	memset(*mem_int,0,sofmi*sizeof(int));

/* For the XYZI integers */

	*mem_XYZI=(XYZI *)realloc(*mem_XYZI, sofmXYZI*sizeof(XYZI));
	if(!mem_XYZI )
	{
		printf( "failed to allocate memory for XYZI integers\n ");
		exit(1);
	}
	memset(*mem_XYZI,0,sofmXYZI*sizeof(XYZI));

	return 1;
}


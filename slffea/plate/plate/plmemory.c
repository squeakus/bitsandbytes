/*
    This utility function allocates the memory for
    the plate finite element program.

        Updated 6/4/00

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
#include "plconst.h"
#include "plstruct.h"

int plMemory( double **mem_double, int sofmf, int **mem_int, int sofmi, MATL **matl,
	int nmat, XYZPhiI **mem_XYZPhiI, int sofmXYZPhiI, MDIM **mem_MDIM, SDIM **mem_SDIM,
	int sofmSDIM, CURVATURE **curve, MOMENT **moment, STRAIN **strain,
	STRESS **stress, int sofmSTRESS )
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

/* For the MOMENT doubles */

	*moment=(MOMENT *)calloc(sofmSTRESS,sizeof(MOMENT));
	if(!moment )
	{
		printf( "failed to allocate memory for moment doubles\n ");
		exit(1);
	}

/* For the STRESS doubles */

	*stress=(STRESS *)calloc(sofmSTRESS,sizeof(STRESS));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}

/* For the CURVATURE doubles */

	*curve=(CURVATURE *)calloc(sofmSTRESS,sizeof(CURVATURE));
	if(!curve )
	{
		printf( "failed to allocate memory for curve doubles\n ");
		exit(1);
	}

/* For the STRAIN doubles */

	*strain=(STRAIN *)calloc(sofmSTRESS,sizeof(STRAIN));
	if(!strain )
	{
		printf( "failed to allocate memory for strain doubles\n ");
		exit(1);
	}

/* For the integers */
	*mem_int=(int *)calloc(sofmi,sizeof(int));
	if(!mem_int )
	{
		printf( "failed to allocate memory for integers\n ");
		exit(1);
	}

/* For the XYZPhiI integers */
	*mem_XYZPhiI=(XYZPhiI *)calloc(sofmXYZPhiI,sizeof(XYZPhiI));
	if(!mem_XYZPhiI )
	{
		printf( "failed to allocate memory for XYZPhiI integers\n ");
		exit(1);
	}

/* For the MDIM doubles */

	*mem_MDIM=(MDIM *)calloc(sofmSDIM,sizeof(MDIM));
	if(!mem_MDIM )
	{
		printf( "failed to allocate memory for MDIM doubles\n ");
		exit(1);
	}

/* For the SDIM doubles */

	*mem_SDIM=(SDIM *)calloc(sofmSDIM,sizeof(SDIM));
	if(!mem_SDIM )
	{
		printf( "failed to allocate memory for SDIM doubles\n ");
		exit(1);
	}

	return 1;
}

int plReGetMemory( double **mem_double, int sofmf, int **mem_int, int sofmi, MATL **matl,
	int nmat, XYZPhiI **mem_XYZPhiI, int sofmXYZPhiI, MDIM **mem_MDIM, SDIM **mem_SDIM,
	int sofmSDIM, CURVATURE **curve, MOMENT **moment, STRAIN **strain,
	STRESS **stress, int sofmSTRESS )
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

/* For the MOMENT doubles */

	*moment=(MOMENT *)realloc(*moment, sofmSTRESS*sizeof(MOMENT));
	if(!moment )
	{
		printf( "failed to allocate memory for moment doubles\n ");
		exit(1);
	}
	memset(*moment,0,sofmSTRESS*sizeof(MOMENT));

/* For the STRESS doubles */

	*stress=(STRESS *)realloc(*stress, sofmSTRESS*sizeof(STRESS));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}
	memset(*stress,0,sofmSTRESS*sizeof(STRESS));

/* For the CURVATURE doubles */

	*curve=(CURVATURE *)realloc(*curve, sofmSTRESS*sizeof(CURVATURE));
	if(!curve )
	{
		printf( "failed to allocate memory for curve doubles\n ");
		exit(1);
	}
	memset(*curve,0,sofmSTRESS*sizeof(CURVATURE));

/* For the STRAIN doubles */

	*strain=(STRAIN *)realloc(*strain, sofmSTRESS*sizeof(STRAIN));
	if(!strain )
	{
		printf( "failed to allocate memory for strain doubles\n ");
		exit(1);
	}
	memset(*strain,0,sofmSTRESS*sizeof(STRAIN));

/* For the integers */
	*mem_int=(int *)realloc(*mem_int, sofmi*sizeof(int));
	if(!mem_int )
	{
		printf( "failed to allocate memory for integers\n ");
		exit(1);
	}
	memset(*mem_int,0,sofmi*sizeof(int));

/* For the XYZPhiI integers */
	*mem_XYZPhiI=(XYZPhiI *)realloc(*mem_XYZPhiI, sofmXYZPhiI*sizeof(XYZPhiI));
	if(!mem_XYZPhiI )
	{
		printf( "failed to allocate memory for XYZPhiI integers\n ");
		exit(1);
	}
	memset(*mem_XYZPhiI,0,sofmXYZPhiI*sizeof(XYZPhiI));

/* For the MDIM doubles */

	*mem_MDIM=(MDIM *)realloc(*mem_MDIM, sofmSDIM*sizeof(MDIM));
	if(!mem_MDIM )
	{
		printf( "failed to allocate memory for MDIM doubles\n ");
		exit(1);
	}
	memset(*mem_MDIM,0,sofmSDIM*sizeof(MDIM));

/* For the SDIM doubles */

	*mem_SDIM=(SDIM *)realloc(*mem_SDIM, sofmSDIM*sizeof(SDIM));
	if(!mem_SDIM )
	{
		printf( "failed to allocate memory for SDIM doubles\n ");
		exit(1);
	}
	memset(*mem_SDIM,0,sofmSDIM*sizeof(SDIM));

	return 1;
}

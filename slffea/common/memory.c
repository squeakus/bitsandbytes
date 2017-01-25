/*
    This utility function allocates the memory for 3-D non-constant strain
    elements(an example of a constant strain element is the
    tetrahedron) for the finite element program.

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
#include <string.h>

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
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#endif
#if WEDGE2
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge2/we2struct.h"
#endif


int Memory( double **mem_double, int sofmf, int **mem_int, int sofmi,
	MATL **matl, int nmat, XYZI **mem_XYZI, int sofmXYZI, SDIM **mem_SDIM,
	int sofmSDIM, STRAIN **strain, STRESS **stress, int sofmSTRESS )
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

/* For the STRESS doubles */

	*stress=(STRESS *)calloc(sofmSTRESS,sizeof(STRESS));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
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

/* For the XYZI integers */

	*mem_XYZI=(XYZI *)calloc(sofmXYZI,sizeof(XYZI));
	if(!mem_XYZI )
	{
		printf( "failed to allocate memory for XYZI integers\n ");
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


int ReGetMemory( double **mem_double, int sofmf, int **mem_int, int sofmi,
	MATL **matl, int nmat, XYZI **mem_XYZI, int sofmXYZI, SDIM **mem_SDIM,
	int sofmSDIM, STRAIN **strain, STRESS **stress, int sofmSTRESS )
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

/* For the STRESS doubles */

	*stress=(STRESS *)realloc(*stress, sofmSTRESS*sizeof(STRESS));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}
	memset(*stress,0,sofmSTRESS*sizeof(STRESS));

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

/* For the XYZI integers */

	*mem_XYZI=(XYZI *)realloc(*mem_XYZI, sofmXYZI*sizeof(XYZI));
	if(!mem_XYZI )
	{
		printf( "failed to allocate memory for XYZI integers\n ");
		exit(1);
	}
	memset(*mem_XYZI,0,sofmXYZI*sizeof(XYZI));

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


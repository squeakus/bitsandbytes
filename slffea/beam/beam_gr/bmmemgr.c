/*
    This utility function allocates the memory for the graphics variables for
    the beam finite element graphics program.

  
   			Last Update 3/16/00

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"

int bmMemory_gr( ICURVATURE **curve_color, IMOMENT **moment_color,
	ISTRAIN **strain_color, ISTRESS **stress_color, int sofmISTRESS)
{
/* Allocate the memory for when first running the program

	Updated 12/7/99

*/

/* For the IMOMENT integers */

	*moment_color=(IMOMENT *)calloc(sofmISTRESS,sizeof(IMOMENT));
	if(!moment_color )
	{
		printf( "failed to allocate memory for moment integers\n ");
		exit(1);
	}

/* For the ISTRESS integers */

	*stress_color=(ISTRESS *)calloc(sofmISTRESS,sizeof(ISTRESS));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}

/* For the ICURVATURE integers */

	*curve_color=(ICURVATURE *)calloc(sofmISTRESS,sizeof(ICURVATURE));
	if(!curve_color )
	{
		printf( "failed to allocate memory for curve integers\n ");
		exit(1);
	}

/* For the ISTRAIN integers */

	*strain_color=(ISTRAIN *)calloc(sofmISTRESS,sizeof(ISTRAIN));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}
	return 1; 
}


int bmMemory2_gr( XYZPhiF **mem_XYZPhiF, int sofmXYZPhiF, XYZF_GR **dist_load_vec,
	int sofmXYZF_GR, QYQZ **dist_load_vec0, int sofmQYQZ)
{

/* For the XYZPhiF doubles */

	*mem_XYZPhiF=(XYZPhiF *)calloc(sofmXYZPhiF,sizeof(XYZPhiF));
	if(!mem_XYZPhiF )
	{
		printf( "failed to allocate memory for XYZPhiF doubles\n ");
		exit(1);
	}

/* For the XYZF_GR doubles */

	*dist_load_vec=(XYZF_GR *)calloc(sofmXYZF_GR,sizeof(XYZF_GR));
	if(!dist_load_vec )
	{
		printf( "failed to allocate memory for XYZF_GR double\n ");
		exit(1);
	}

/* For the QYQZ doubles */

	*dist_load_vec0=(QYQZ *)calloc(sofmQYQZ,sizeof(QYQZ));
	if(!dist_load_vec0 )
	{
		printf( "failed to allocate memory for QYQZ double\n ");
		exit(1);
	}

	return 1; 
}

int bmReGetMemory_gr( ICURVATURE **curve_color, IMOMENT **moment_color,
	ISTRAIN **strain_color, ISTRESS **stress_color, int sofmISTRESS)
{
/* Allocate the memory for when first running the program

	Updated 3/14/00

*/

/* For the IMOMENT integers */

	*moment_color=(IMOMENT *)realloc(*moment_color, sofmISTRESS*sizeof(IMOMENT));
	if(!moment_color )
	{
		printf( "failed to allocate memory for moment integers\n ");
		exit(1);
	}
	memset(*moment_color,0,sofmISTRESS*sizeof(IMOMENT));

/* For the ISTRESS integers */

	*stress_color=(ISTRESS *)realloc(*stress_color, sofmISTRESS*sizeof(ISTRESS));
	if(!stress_color )
	{
		printf( "failed to allocate memory for stress integers\n ");
		exit(1);
	}
	memset(*stress_color,0,sofmISTRESS*sizeof(ISTRESS));

/* For the ICURVATURE integers */

	*curve_color=(ICURVATURE *)realloc(*curve_color, sofmISTRESS*sizeof(ICURVATURE));
	if(!curve_color )
	{
		printf( "failed to allocate memory for curve integers\n ");
		exit(1);
	}
	memset(*curve_color,0,sofmISTRESS*sizeof(ICURVATURE));

/* For the ISTRAIN integers */

	*strain_color=(ISTRAIN *)realloc(*strain_color, sofmISTRESS*sizeof(ISTRAIN));
	if(!strain_color )
	{
		printf( "failed to allocate memory for strain integers\n ");
		exit(1);
	}
	memset(*strain_color,0,sofmISTRESS*sizeof(ISTRAIN));

	return 1; 
}


int bmReGetMemory2_gr( XYZPhiF **mem_XYZPhiF, int sofmXYZPhiF, XYZF_GR **dist_load_vec,
	int sofmXYZF_GR, QYQZ **dist_load_vec0, int sofmQYQZ)
{

/* For the XYZPhiF doubles */

	*mem_XYZPhiF=(XYZPhiF *)realloc(*mem_XYZPhiF, sofmXYZPhiF*sizeof(XYZPhiF));
	if(!mem_XYZPhiF )
	{
		printf( "failed to allocate memory for XYZPhiF doubles\n ");
		exit(1);
	}
	memset(*mem_XYZPhiF,0,sofmXYZPhiF*sizeof(XYZPhiF));

/* For the XYZF_GR doubles */

	*dist_load_vec=(XYZF_GR *)realloc(*dist_load_vec, sofmXYZF_GR*sizeof(XYZF_GR));
	if(!dist_load_vec )
	{
		printf( "failed to allocate memory for XYZF_GR double\n ");
		exit(1);
	}
	memset(*dist_load_vec,0,sofmXYZF_GR*sizeof(XYZF_GR));

/* For the QYQZ doubles */

	*dist_load_vec0=(QYQZ *)realloc(*dist_load_vec0, sofmQYQZ*sizeof(QYQZ));
	if(!dist_load_vec0 )
	{
		printf( "failed to allocate memory for QYQZ double\n ");
		exit(1);
	}
	memset(*dist_load_vec0,0,sofmQYQZ*sizeof(QYQZ));

	return 1; 
}

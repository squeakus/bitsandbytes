/*
    This program Calculates the normal vectors of a mesh
    for triangle elements.
  
  		Last Update 9/16/06

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
#if TRI1
#include "../tri/trconst.h"
#include "trstrcgr.h"
#endif
#if PLATE1
#include "../../plate/plate/plconst.h"
#include "../../plate/plate_gr/plstrcgr.h"
#endif

extern int numel;

int normcrossX(double *, double *, double *);

int trnormal_vectors(int *connecter, double *coord, NORM *norm )
{
	int i, i2, j, k, sdof_el[npel3*nsd], ii, check, counter, node;
	int l,m,n;
	double coord_el[npel3*3];
	double d1[3], d2[3], norm_temp[3];

	for( k = 0; k < numel; ++k )
	{
		for( j = 0; j < npel3; ++j )
		{

/* Calculate element degrees of freedom */

			node = *(connecter+npel3*k+j);
			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

/* Calculate local coordinates */

			*(coord_el+3*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+3*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+3*j+2)=*(coord+*(sdof_el+nsd*j+2));

			/*printf( "%9.5f %9.5f %9.5f \n",*(coord_el+3*j),
				*(coord_el+3*j+1),*(coord_el+3*j+2));*/
		}

/* Calculate normal vectors */

/* Triangle face 0 */

		*(d1)=*(coord_el+6)-*(coord_el+3);
		*(d1+1)=*(coord_el+7)-*(coord_el+4);
		*(d1+2)=*(coord_el+8)-*(coord_el+5);
		*(d2)=*(coord_el)-*(coord_el+3);
		*(d2+1)=*(coord_el+1)-*(coord_el+4);
		*(d2+2)=*(coord_el+2)-*(coord_el+5);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[0].x = *(norm_temp);
		norm[k].face[0].y = *(norm_temp+1);
		norm[k].face[0].z = *(norm_temp+2);

	}
	return 1;
}



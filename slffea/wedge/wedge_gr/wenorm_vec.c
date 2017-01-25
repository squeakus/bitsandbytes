/*
    This program Calculates the normal vectors of a mesh
    for wedge elements.
  
  		Last Update 9/25/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if WEDGE1
#include "../wedge/weconst.h"
#include "westrcgr.h"
#endif
#if SHELL1
#include "../../shell/shell/shconst.h"
#include "../../shell/shell_gr/shstrcgr.h"
#endif

extern int numel, numnp;

int normcrossX(double *, double *, double *);

int wenormal_vectors(int *connecter, double *coord, NORM *norm )
{
	int i, i2, j, k, sdof_el[npel6*nsd], ii, check, counter, node;
	int l,m,n;
	double coord_el[npel6*3];
	double d1[3], d2[3], norm_temp[3];

	for( k = 0; k < numel; ++k )
	{
#if WEDGE1
		for( j = 0; j < npel6; ++j )
		{

/* Calculate element degrees of freedom */

			node = *(connecter+npel6*k+j);
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
#endif

#if SHELL1
		for( j = 0; j < npell3; ++j )
		{

/* Calculate element degrees of freedom */

			node = *(connecter+npell3*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(sdof_el+nsd*npell3+nsd*j)=nsd*(node+numnp);
			*(sdof_el+nsd*npell3+nsd*j+1)=nsd*(node+numnp)+1;
			*(sdof_el+nsd*npell3+nsd*j+2)=nsd*(node+numnp)+2;

/* Calculate local coordinates */

			*(coord_el+3*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+3*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+3*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el+3*npell3+3*j)=
				*(coord+*(sdof_el+nsd*npell3+nsd*j));
			*(coord_el+3*npell3+3*j+1)=
				*(coord+*(sdof_el+nsd*npell3+nsd*j+1));
			*(coord_el+3*npell3+3*j+2)=
				*(coord+*(sdof_el+nsd*npell3+nsd*j+2));

			/*printf( "%9.5f %9.5f %9.5f \n",*(coord_el+3*j),
				*(coord_el+3*j+1),*(coord_el+3*j+2));*/
		}
#endif


/* Calculate normal vectors */

/* Triangle face 0 */

		*(d1)=*(coord_el)-*(coord_el+3);
		*(d1+1)=*(coord_el+1)-*(coord_el+4);
		*(d1+2)=*(coord_el+2)-*(coord_el+5);
		*(d2)=*(coord_el+6)-*(coord_el+3);
		*(d2+1)=*(coord_el+7)-*(coord_el+4);
		*(d2+2)=*(coord_el+8)-*(coord_el+5);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[0].x = *(norm_temp);
		norm[k].face[0].y = *(norm_temp+1);
		norm[k].face[0].z = *(norm_temp+2);

/* Triangle face 1 */

		*(d1)=*(coord_el+12)-*(coord_el+3);
		*(d1+1)=*(coord_el+13)-*(coord_el+4);
		*(d1+2)=*(coord_el+14)-*(coord_el+5);
		*(d2)=*(coord_el)-*(coord_el+3);
		*(d2+1)=*(coord_el+1)-*(coord_el+4);
		*(d2+2)=*(coord_el+2)-*(coord_el+5);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[1].x = *(norm_temp);
		norm[k].face[1].y = *(norm_temp+1);
		norm[k].face[1].z = *(norm_temp+2);

/* Triangle face 2 */

		*(d1)=*(coord_el)-*(coord_el+9);
		*(d1+1)=*(coord_el+1)-*(coord_el+10);
		*(d1+2)=*(coord_el+2)-*(coord_el+11);
		*(d2)=*(coord_el+12)-*(coord_el+9);
		*(d2+1)=*(coord_el+13)-*(coord_el+10);
		*(d2+2)=*(coord_el+14)-*(coord_el+11);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[2].x = *(norm_temp);
		norm[k].face[2].y = *(norm_temp+1);
		norm[k].face[2].z = *(norm_temp+2);

/* Triangle face 3 */

		*(d1)=*(coord_el+6)-*(coord_el+3);
		*(d1+1)=*(coord_el+7)-*(coord_el+4);
		*(d1+2)=*(coord_el+8)-*(coord_el+5);
		*(d2)=*(coord_el+12)-*(coord_el+3);
		*(d2+1)=*(coord_el+13)-*(coord_el+4);
		*(d2+2)=*(coord_el+14)-*(coord_el+5);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[3].x = *(norm_temp);
		norm[k].face[3].y = *(norm_temp+1);
		norm[k].face[3].z = *(norm_temp+2);

/* Triangle face 4 */

		*(d1)=*(coord_el+12)-*(coord_el+15);
		*(d1+1)=*(coord_el+13)-*(coord_el+16);
		*(d1+2)=*(coord_el+14)-*(coord_el+17);
		*(d2)=*(coord_el+6)-*(coord_el+15);
		*(d2+1)=*(coord_el+7)-*(coord_el+16);
		*(d2+2)=*(coord_el+8)-*(coord_el+17);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[4].x = *(norm_temp);
		norm[k].face[4].y = *(norm_temp+1);
		norm[k].face[4].z = *(norm_temp+2);

/* Triangle face 5 */

		*(d1)=*(coord_el+6)-*(coord_el+15);
		*(d1+1)=*(coord_el+7)-*(coord_el+16);
		*(d1+2)=*(coord_el+8)-*(coord_el+17);
		*(d2)=*(coord_el+9)-*(coord_el+15);
		*(d2+1)=*(coord_el+10)-*(coord_el+16);
		*(d2+2)=*(coord_el+11)-*(coord_el+17);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[5].x = *(norm_temp);
		norm[k].face[5].y = *(norm_temp+1);
		norm[k].face[5].z = *(norm_temp+2);

/* Triangle face 6 */

		*(d1)=*(coord_el+9)-*(coord_el);
		*(d1+1)=*(coord_el+10)-*(coord_el+1);
		*(d1+2)=*(coord_el+11)-*(coord_el+2);
		*(d2)=*(coord_el+6)-*(coord_el);
		*(d2+1)=*(coord_el+7)-*(coord_el+1);
		*(d2+2)=*(coord_el+8)-*(coord_el+2);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[6].x = *(norm_temp);
		norm[k].face[6].y = *(norm_temp+1);
		norm[k].face[6].z = *(norm_temp+2);

/* Triangle face 7 */

		*(d1)=*(coord_el+15)-*(coord_el+12);
		*(d1+1)=*(coord_el+16)-*(coord_el+13);
		*(d1+2)=*(coord_el+17)-*(coord_el+14);
		*(d2)=*(coord_el+9)-*(coord_el+12);
		*(d2+1)=*(coord_el+10)-*(coord_el+13);
		*(d2+2)=*(coord_el+11)-*(coord_el+14);
		normcrossX(d1, d2, norm_temp);
		norm[k].face[7].x = *(norm_temp);
		norm[k].face[7].y = *(norm_temp+1);
		norm[k].face[7].z = *(norm_temp+2);

	}
	return 1;
}



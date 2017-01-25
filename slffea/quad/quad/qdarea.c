/*
    This utility function calculates the Area of each
    quad element using shape functions and gaussian
    integration.

    I used to have a lot of code relating to finding the local axis
    at each node because this has been put in the subroutine 
    "local_vectors_old".  It should be noted though, that the method
    applied is based on that of the shell element, and can result
    in local vectors which are very different.  This then results
    in rotations that cause the coordinates to rotate in an
    incompatable way, resulting in negative areas.  So it is better
    to calculate local vectors based on the that prescribed in
    the subroutine "local_vectors".

                Updated 12/21/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

*/

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qdconst.h"

extern int numel, flag_3D, sof, local_vec_flag;
extern double shg[sosh], shl[sosh], w[num_int];

int qdshg( double *, int, double *, double *, double *);

int local_vectors_old( int *, double *, double * );

int local_vectors( int *, double *, double * );

int qdArea( int *connect, double *coord, double *Area)
{
	int i, i2, j, k, node, check, dum;
	double fdum1, fdum2, fdum3, fdum4;
	double rotate[npel*nsd2*nsd];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double det[num_int];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd], vec_dum4[nsd],
		xp0[npel], xp1[npel], xp2[npel], yp0[npel], yp1[npel], yp2[npel],
		zp0[npel], zp1[npel], zp2[npel];
	double *local_xyz;

	local_xyz = (double *)calloc(numel*npel*nsdsq, sizeof(double));


	if(local_vec_flag)
	{
	    check = local_vectors( connect, coord, local_xyz );
	    if(!check) printf( " Problems with local_vectors \n");
	}
	else
	{
	    check = local_vectors_old( connect, coord, local_xyz );
	    if(!check) printf( " Problems with local_vectors_old \n");
	}


	for( k = 0; k < numel; ++k )
	{

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node=*(connect+npel*k+j);

                        *(coord_el+nsd*j)=*(coord+nsd*node);
                        *(coord_el+nsd*j+1)=*(coord+nsd*node+1);
                        *(coord_el+nsd*j+2)=*(coord+nsd*node+2);

			*(coord_el_trans+j)=*(coord+nsd*node);
			*(coord_el_trans+npel*1+j)=*(coord+nsd*node+1);
			*(coord_el_trans+npel*2+j)=*(coord+nsd*node+2);
		}

		if(!flag_3D)
		{
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel + j) = *(coord_el_trans + npel*1 + j);
		    }
		}
		else
		{

		    if(!local_vec_flag)
		    {
/* This is the first method. */

			for( j = 0; j < npel; ++j )
			{
			    *(rotate + j*nsd2*nsd) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j);
			    *(rotate + j*nsd2*nsd + 1) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 1);
			    *(rotate + j*nsd2*nsd + 2) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 2);
			    *(rotate + j*nsd2*nsd + 3) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 3);
			    *(rotate + j*nsd2*nsd + 4) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 4);
			    *(rotate + j*nsd2*nsd + 5) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 5);

/* Put coord_el into local coordinates */

			    check = matX( (coord_el_local+nsd2*j), (rotate + j*nsd2*nsd),
				(coord_el+nsd*j), nsd2, 1, nsd);
			    if(!check) printf( "Problems with  matX \n");
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}

/* The code below does the same as the matrix multiplication above to calculate the local
   coordinates and is clearer but isn't as efficient.  If the code below is used, make sure
   to comment out both the matrix multiplication above as well as the setting of
   coord_el_trans.

			dum = nsd*npel;
			check = rotXmat3(coord_el_local, rotate, coord_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat3 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}
*/
		    }
		    else
		    {
/* This is the second method. */

			*(rotate)     = *(local_xyz + nsdsq*k);
			*(rotate + 1) = *(local_xyz + nsdsq*k + 1);
			*(rotate + 2) = *(local_xyz + nsdsq*k + 2);
			*(rotate + 3) = *(local_xyz + nsdsq*k + 3);
			*(rotate + 4) = *(local_xyz + nsdsq*k + 4);
			*(rotate + 5) = *(local_xyz + nsdsq*k + 5);

			dum = nsd*npel;
			check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat2 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}
		    }

		}

/* Assembly of the shg matrix for each integration point */

		check=qdshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with qdshg \n");

/* Calculate the Area from determinant of the Jacobian */

		for( j = 0; j < num_int; ++j )
		{
			*(Area + k) += *(w+j)*(*(det+j));
		}
	}

	free(local_xyz);

	return 1;
}

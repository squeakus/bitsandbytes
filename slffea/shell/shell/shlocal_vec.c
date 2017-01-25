/*
    This utility function calculates the local x, y, z fiber basis
    vectors for each node for a finite element program which does
    analysis on a shell element.  It also calculates the nodal lamina
    reference vector.

		Updated 10/19/06

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
#include <math.h>
#include "shconst.h"
#include "shstruct.h"

extern int analysis_flag, dof, sdof, integ_flag, doubly_curved_flag,
	numel, numnp, neqn, sof;

int normcrossX(double *, double *, double *);

int matX( double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int shlocal_vectors( double *coord, double *lamina_ref, double *fiber_xyz ) 
{
	int i, i1, i2, i3, i4, i5, j, k, dof_el[neqel20], sdof_el[npel8*nsd];
	int check, counter, node;
	double fdum1, fdum2, fdum3, fdum4;
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd];

	for( i = 0; i < numnp; ++i )
	{

/* Create the coord_bar and coord_hat vector for one element */

		*(lamina_ref + nsd*i) =
			.5*( *(coord + nsd*(i+numnp))*(1.0+zeta) + *(coord + nsd*i)*(1.0-zeta));
		*(lamina_ref + nsd*i + 1) =
			.5*( *(coord + nsd*(i+numnp) + 1)*(1.0+zeta) + *(coord + nsd*i + 1)*(1.0-zeta));
		*(lamina_ref + nsd*i + 2) =
			.5*( *(coord + nsd*(i+numnp) + 2)*(1.0+zeta) + *(coord + nsd*i + 2)*(1.0-zeta));

		*(local_z) = *(coord + nsd*(i+numnp)) - *(coord + nsd*i);
		*(local_z + 1) = *(coord + nsd*(i+numnp) + 1) - *(coord + nsd*i + 1);
		*(local_z + 2) = *(coord + nsd*(i+numnp) + 2) - *(coord + nsd*i + 2);

		fdum1=fabs(*(local_z));
		fdum2=fabs(*(local_z+1));
		fdum3=fabs(*(local_z+2));
		fdum4=sqrt(fdum1*fdum1+fdum2*fdum2+fdum3*fdum3);
		*(local_z) /= fdum4;
		*(local_z+1) /= fdum4;
		*(local_z+2) /= fdum4;

/*
   Calculating rotation matrix for the fiber q[i,j] matrix.

   The algorithm below is taken from "The Finite Element Method" by Thomas Hughes,
   page 388.  The goal is to find the local shell coordinates which come closest
   to the global x, y, z coordinates.  In the algorithm below, vec_dum is set to either
   the global x, y, or z basis vector based on the one of the 2 smaller components of
   xl.hat.  xl.hat itself is the local z direction of that node.  Once set, the cross
   product of local z and vec_dum produces the local y fiber direction.  This local y is
   then crossed with local z to produce local x.
*/
		memset(vec_dum,0,nsd*sof);
		i2=1;
		if( fdum1 > fdum3)
		{
			fdum3=fdum1;
			i2=2;
		}
		if( fdum2 > fdum3) i2=3;
		*(vec_dum+(i2-1))=1.0;

/* Calculate the local y basis vector = local z X vec_dum */
		check = normcrossX( local_z, vec_dum, local_y);
		if(!check) printf( "Problems with normcrossX \n");

/* Calculate the local x basis vector = local y X local z */
		check = normcrossX( local_y, local_z, local_x );
		if(!check) printf( "Problems with normcrossX \n");

/* Create the global rotation matrix */

		for( i1 = 0; i1 < nsd; ++i1 )
		{
		    *(fiber_xyz + nsdsq*i + i1) = *(local_x + i1);
		    *(fiber_xyz + nsdsq*i + 1*nsd + i1) = *(local_y + i1);
		    *(fiber_xyz + nsdsq*i + 2*nsd + i1) = *(local_z + i1);
		}

	}

	return 1;
}


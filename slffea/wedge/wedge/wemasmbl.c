/*
    This utility function assembles either the lumped sum global
    diagonal Mass matrix or, in the case of consistent mass, all
    the element mass matrices.  This is for a finite element program
    which does analysis on a wedge element.  It is for
    modal analysis.

		Updated 8/22/06

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
#include "weconst.h"
#include "westruct.h"

extern int dof, numel, numnp, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Vol0;
extern int consistent_mass_flag, consistent_mass_store, lumped_mass_flag;

int matXT(double *, double *, double *, int, int, int);

int wedgeB_mass(double *,double *);

int weshg_mass( double *, int, double *, double *);

int weMassemble(int *connect, double *coord, int *el_matl, int *id,
	double *mass, MATL *matl) 
	
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
	double B_mass[MsoB], B2_mass[MsoB];
	double M_temp[neqlsq], M_el[neqlsq];
	double coord_el_trans[neqel];
	double det[num_int], volume_el, wXdet;
	double mass_el[neqel];

/*      initialize all variables  */

	memcpy(shg,shl,sosh*sizeof(double));

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		rho = matl[matl_num].rho;
		volume_el = 0.0;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* The call to weshg_mass is only for calculating the determinent */

		check = weshg_mass(det, k, shg, coord_el_trans);
		if(!check) printf( "Problems with weshg_mass \n");

#if 0
		for( i1 = 0; i1 < num_int; ++i1 )
		{
		    for( i2 = 0; i2 < npel; ++i2 )
		    {
			printf("%10.6f ",*(shl+npel*(nsd+1)*i1 + npel*(nsd) + i2));
		    }
		    printf(" \n");
		}
		printf(" \n");
#endif

/* The loop over j below calculates the 6 points of numerical integration
   for several quantities */

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B_mass,0,MsoB*sof);
		    memset(B2_mass,0,MsoB*sof);
		    memset(M_temp,0,neqlsq*sof);

/* Assembly of the B matrix for mass */

		    check = wedgeB_mass((shg+npel*(nsd+1)*j + npel*(nsd)),B_mass);
		    if(!check) printf( "Problems with wedgeB_mass \n");

#if 0
		    for( i1 = 0; i1 < nsd; ++i1 )
		    {
			for( i2 = 0; i2 < neqel; ++i2 )
			{
				printf("%9.6f ",*(B_mass+neqel*i1+i2));
			}
			printf(" \n");
		    }
		    printf(" \n");
#endif

		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

		    check=matXT(M_temp, B_mass, B2_mass, neqel, neqel, nsd);
		    if(!check) printf( "Problems with matXT \n");

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in 
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		    wXdet = 0.5*(*(w+j))*(*(det+j));

/* Calculate the Volume from determinant of the Jacobian */

		    volume_el += wXdet;

		    fdum =  rho*wXdet;
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(M_el+i2) += *(M_temp+i2)*fdum;
		    }
		}

		/* printf("This is 3 X Volume %10.6f for element %4d\n",3.0*volume_el,k);*/

		if(lumped_mass_flag)
		{

/* Creating the diagonal lumped mass Matrix */

		    fdum = 0.0;
		    for( i2 = 0; i2 < neqel; ++i2 )
		    {   
			/*printf("This is mass_el for el %3d",k);*/
			for( i3 = 0; i3 < neqel; ++i3 )
			{
			    *(mass_el+i2) += *(M_el+neqel*i2+i3);
			}
			/*printf("%9.6f\n\n",*(mass_el+i2));*/
			fdum += *(mass_el+i2);
		    }   
		    /* printf("This is Volume2 %9.6f\n\n",fdum);*/

		    for( j = 0; j < neqel; ++j )
		    {   
			*(mass+*(dof_el+j)) += *(mass_el + j);
		    }
		}

		if(consistent_mass_flag)
		{

/* Storing all the element mass matrices */

		    for( j = 0; j < neqlsq; ++j )
		    {   
			*(mass + neqlsq*k + j) = *(M_el + j);
		    }
		}
	}

	if(lumped_mass_flag)
	{
/* Contract the global mass matrix using the id array only if lumped
   mass is used. */

	    counter = 0;
	    for( i = 0; i < dof ; ++i )
	    {
		/* printf("%5d  %16.8e\n", i, *(mass+i));*/
		if( *(id + i ) > -1 )
		{
		    *(mass + counter ) = *(mass + i );
		    ++counter;
		}
	    }
	}

	return 1;
}

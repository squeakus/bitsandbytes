/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a brick.  It is for modal analysis.

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
#include "brconst.h"
#include "brstruct.h"

extern int numel, numnp, dof, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Vol0;
extern int consistent_mass_flag, consistent_mass_store;

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int brickB_mass(double *,double *);

int brshg_mass( double *, int, double *, double *);

int brMassPassemble(int *connect, double *coord, int *el_matl, double *mass,
	MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
	double B_mass[MsoB], B2_mass[MsoB];
	double M_temp[neqlsq], M_el[neqlsq];
	double U_el[neqel];
	double coord_el_trans[neqel];
	double det[num_int];
	double P_el[neqel];

	memset(P_global,0,dof*sof);

	memcpy(shg,shl,sosh*sizeof(double));

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (mass + k*neqlsq), U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}
	    }
	}
	else
	{

/* Assemble P matrix by re-deriving element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {
		matl_num = *(el_matl+k);
		rho = matl[matl_num].rho;

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

/* The call to brshg_mass is only for calculating the determinent */

		check = brshg_mass(det, k, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg_mass \n");

#if 0
		for( i1 = 0; i1 < num_int; ++i1 )
		{
		    for( i2 = 0; i2 < npel; ++i2 )
		    {
			printf("%10.6f ",*(shl+npel*(nsd+1)*i1 + npel*(nsd) + i2));
		    }
		    printf(" \n ");
		}
		printf(" \n ");
#endif

/* The loop over j below calculates the 8 points of the gaussian integration
   for several quantities */

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B_mass,0,MsoB*sof);
		    memset(B2_mass,0,MsoB*sof);
		    memset(M_temp,0,neqlsq*sof);

/* Assembly of the B matrix for mass */

		    check = brickB_mass((shg+npel*(nsd+1)*j + npel*(nsd)),B_mass);
		    if(!check) printf( "Problems with brickB_mass \n");

#if 0
		    for( i1 = 0; i1 < nsd; ++i1 )
		    {
			for( i2 = 0; i2 < neqel; ++i2 )
			{
				printf("%9.6f ",*(B_mass+neqel*i1+i2));
			}
			printf(" \n ");
		    }
		    printf(" \n ");
#endif
		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

		    check=matXT(M_temp, B_mass, B2_mass, neqel, neqel, nsd);
		    if(!check) printf( "Problems with matXT \n");

		    fdum =  rho*(*(w+j))*(*(det+j));
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(M_el+i2) += *(M_temp+i2)*fdum;
		    }
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, M_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}

	    }
	}

	return 1;
}

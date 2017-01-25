/*
    This utility function assembles the Mass and force matrix for a finite
    element program which does analysis on a wedge which can behave
    nonlinearly.  The mass is used both in the dynamic relaxation
    method as well as the conjugate gradient method.   It is not really
    needed in the conjugate gradient method except to act as a
    preconditioner.

    Note that the mass matrix is made up of the diagonal of the stiffness
    matrix.

    Currently, there are quantities at 1/2 time used to calculate
    the stiffness.  Because coordh and coord are the same at the beginning
    of the calculation, there is no difference in which is used.  But
    because there is a possibility that I will recalculate the mass
    during the main calculation loop which has coordinate updating,
    I will leave the code as it is.  If this does happen, then I will
    need to comment out the lines dealing with force.

    Note that I do not use Bh for DB in we2Passemble when the conjugate
    gradient method is used.  This is because I want to keep things consistent
    with weConjPassemble.

		Updated 11/25/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../wedge/weconst.h"
#include "../wedge/westruct.h"

extern int numel, numnp, dof, sof;
extern double shg[sosh], shgh[sosh], shl[sosh], w[num_int], *Vol0;

int cubic( double *);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int wedgeB( double *,double *);

int weshg( double *, int, double *, double *, double *);

int westress_shg( double *, int, double *, double *, double * );

int weFMassemble(int *connect, double *coord, double *coordh, int *el_matl,
	double *force, double *mass, MATL *matl, double *U)
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, surf_el_counter, surface_el_flag;
	int matl_num;
	double Emod, Pois, G;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], Bh[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double force_el[neqel], U_el[neqel];
	double coord_el[neqel], coordh_el[neqel];
	double coord_el_trans[neqel], coordh_el_trans[neqel];
	double stress_el[sdim], strain_el[sdim], invariant[nsd],
		yzsq, zxsq, xysq, xxyy;
	double det[num_int], deth[num_int], volume_el, wXdet;
	double mass_el[neqel];

/* initialize all variables  */
	memset(mass,0,dof*sof); 

	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		volume_el = 0.0;

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));
		mu = Emod/(1.0+Pois)/2.0;

		D11 = lamda+2.0*mu;
		D12 = lamda;
		D13 = lamda;
		D21 = lamda;
		D22 = lamda+2.0*mu;
		D23 = lamda;
		D31 = lamda;
		D32 = lamda;
		D33 = lamda+2.0*mu;

		G = mu;

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;

/* Create the coord and coord_trans vector for one element */

			*(coord_el+nsd*j) = *(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1) = *(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2) = *(coord+*(sdof_el+nsd*j+2));
			*(coord_el_trans+j) = *(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

/* Create the coordh and coordh_trans vector for one element */

			*(coordh_el+nsd*j) = *(coordh+*(sdof_el+nsd*j));
			*(coordh_el+nsd*j+1) = *(coordh+*(sdof_el+nsd*j+1));
			*(coordh_el+nsd*j+2) = *(coordh+*(sdof_el+nsd*j+2));
			*(coordh_el_trans+j) = *(coordh+nsd*node);
			*(coordh_el_trans+npel*1+j) = *(coordh+nsd*node+1);
			*(coordh_el_trans+npel*2+j) = *(coordh+nsd*node+2);
		}


/* Assembly of the shg matrix for each integration point at 1/2 time */

		check=weshg(deth, k, shl, shgh, coordh_el_trans);
		if(!check) printf( "Problems with weshg \n");

/* Assembly of the shg matrix for each integration point at full time */

		check=weshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with weshg \n");


/* The loop over j below calculates the 6 points of numerical integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);
		memset(force_el,0,neqel*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(Bh,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_temp,0,neqlsq*sof);

/* Assembly of the B matrix at 1/2 time */

		    check = wedgeB((shgh+npel*(nsd+1)*j),Bh);
		    if(!check) printf( "Problems with wedgeB \n");

/* Assembly of the B matrix at full time */

		    check = wedgeB((shg+npel*(nsd+1)*j),B);
		    if(!check) printf( "Problems with wedgeB \n");

		    for( i1 = 0; i1 < neqel; ++i1 )
		    {
			*(DB+i1) = *(Bh+i1)*D11+
				*(Bh+neqel*1+i1)*D12+
				*(Bh+neqel*2+i1)*D13;
			*(DB+neqel*1+i1) = *(Bh+i1)*D21+
				*(Bh+neqel*1+i1)*D22+
				*(Bh+neqel*2+i1)*D23;
			*(DB+neqel*2+i1) = *(Bh+i1)*D31+
				*(Bh+neqel*1+i1)*D32+
				*(Bh+neqel*2+i1)*D33;
			*(DB+neqel*3+i1) = *(Bh+neqel*3+i1)*G;
			*(DB+neqel*4+i1) = *(Bh+neqel*4+i1)*G;
			*(DB+neqel*5+i1) = *(Bh+neqel*5+i1)*G; 
		    }

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in 
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		    wXdet = 0.5*(*(w+j))*(*(det+j));

		    volume_el += wXdet;

		    check=matXT(K_temp, B, DB, neqel, neqel, sdim);
		    if(!check) printf( "Problems with matXT \n");
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(K_el+i2) += *(K_temp+i2)*wXdet;
		    }

		}

		/*printf("element %d %14.6e\n",k,volume_el);*/

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(force_el, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

/* Compute the equivalant nodal forces based on prescribed displacements */

		for( j = 0; j < neqel; ++j )
		{
			*(force + *(dof_el+j)) -= *(force_el + j);
		}

/* Creating the mass Matrix */
		for( i3 = 0; i3 < neqel; ++i3 )
		{
		    *(mass_el+i3) = 100.0*(*(K_el+neqel*i3+i3));
		}
		for( j = 0; j < npel; ++j )
		{
		    *(mass+*(dof_el+ndof*j)) += *(mass_el + ndof*j);
		    *(mass+*(dof_el+ndof*j+1)) += *(mass_el + ndof*j + 1);
		    *(mass+*(dof_el+ndof*j+2)) += *(mass_el + ndof*j + 2);
		}
	}
	/*for( i = 0; i < dof ; ++i )
	{
		printf( " force %4d %16.4e \n",i,*(force+i));
	}*/

	return 1;
}


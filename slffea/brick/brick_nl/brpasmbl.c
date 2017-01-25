/*
    This utility function assembles the P matrix for a finite 
    element program which does analysis on a brick which can behave
    nonlinearly.  The equations are solved using dynamic
    relaxation.   

		Updated 11/25/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../brick/brconst.h"
#include "../brick/brstruct.h"

extern int numel, numnp, dof, sof;
extern double shg[sosh], shgh[sosh], shl[sosh], w[num_int], *Vol0;
extern int Passemble_flag;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int brickB(double *,double *);

int brshg( double *, int, double *, double *, double *);

int brPassemble(int *connect, double *coord, double *coordh, int *el_matl,
	MATL *matl, double *P_global, STRESS *stress, double *dU)

{
	int i,i1,j,k,dof_el[npel*ndof],sdof_el[npel*nsd];
	int check, node;
	int matl_num;
	double Emod, G, K, Pois;
	double D11, D12, D13, D21, D22, D23, D31, D32, D33;
	double lamda, mu, hydro;
	double B[soB], Bh[soB], stressd[sdim];
	double P_el[neqel], P_temp[neqel];
	double dU_el[neqel], domega_el[3], coord_el[npel*nsd], coordh_el[npel*nsd],
		coord_el_trans[npel*nsd], coordh_el_trans[npel*nsd];
	double stress_el[sdim],  dstress_el[sdim], dstrain_el[sdim];
	double det[num_int], deth[num_int], wXdet;
	int i2,i2m1,i2m2;

	memset(P_global,0,dof*sof);

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;

		K=Emod/(1.0-2*Pois)/3.0;
		G=Emod/(1.0+Pois)/2.0;

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));
		mu = Emod/(1.0+Pois)/2.0;
		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

		D11 = lamda+2.0*mu;
		D12 = lamda;
		D13 = lamda;
		D21 = lamda;
		D22 = lamda+2.0*mu;
		D23 = lamda;
		D31 = lamda;
		D32 = lamda;
		D33 = lamda+2.0*mu;

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);
			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;

/* Create the dU_el vector for one element */

			*(dU_el+ndof*j) = *(dU+ndof*node);
			*(dU_el+ndof*j+1) = *(dU+ndof*node+1);
			*(dU_el+ndof*j+2) = *(dU+ndof*node+2);

/* Create the coord vector and coordh_trans for one element */

			*(coord_el+nsd*j) = *(coord+nsd*node);
			*(coord_el+nsd*j+1) = *(coord+nsd*node+1);
			*(coord_el+nsd*j+2) = *(coord+nsd*node+2);
			*(coord_el_trans+j) = *(coord+nsd*node);
			*(coord_el_trans+npel*1+j) = *(coord+nsd*node+1);
			*(coord_el_trans+npel*2+j) = *(coord+nsd*node+2);

/* Create the coordh and coordh_trans vector for one element */

			*(coordh_el+nsd*j) = *(coordh+nsd*node);
			*(coordh_el+nsd*j+1) = *(coordh+nsd*node+1);
			*(coordh_el+nsd*j+2) = *(coordh+nsd*node+2);
			*(coordh_el_trans+j) = *(coordh+nsd*node);
			*(coordh_el_trans+npel*1+j) = *(coordh+nsd*node+1);
			*(coordh_el_trans+npel*2+j) = *(coordh+nsd*node+2);
		}


/* Assembly of the shgh matrix for each integration point at 1/2 time */

		check = brshg(deth, k, shl, shgh, coordh_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* Assembly of the shg matrix for each integration point at full time */

		check = brshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* The loop over j below calculates the 8 points of the gaussian integration 
   for several quantities */

/* Initialize P_el */
		memset(P_el,0,neqel*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B,0,soB*sof);
		    memset(Bh,0,soB*sof);

/* Assembly of the Bh matrix at 1/2 time */

		    check = brickB((shgh+npel*(nsd+1)*j),Bh);
		    if(!check) printf( "Problems with brickB \n");

		    /*for( i2 = 0; i2 < sdim; ++i2 )
		    {
			for( i1 = 0; i1 < 8; ++i1 )
			{
				printf("%6.4f ",*(Bh+neqel*i2+i1));
			}
			printf("\n");
		    }*/


/* Assembly of the B matrix at full time */

		    check = brickB((shg+npel*(nsd+1)*j),B);
		    if(!check) printf( "Problems with brickB \n");

/* Calculation of the incremental strain at 1/2 time */

		    check=matX(dstrain_el,Bh,dU_el, sdim, 1, neqel);
		    if(!check) printf( "Problems with matX \n");

/* Assembly of domega_el(rotation) of the element at 1/2 time = Bh*dU_el */

/* First, Bh is modified slightly before the calculaiton is performed */

		    for( i = 0; i < npel; ++i )
		    {
			i2      = ndof*i+2;
			i2m1    = i2-1;
			i2m2    = i2-2;

			*(Bh+neqel*3+i2m1) *= -1.0;
			*(Bh+neqel*4+i2m2) *= -1.0;
			*(Bh+neqel*5+i2) *= -1.0;
		    }

		    check=matX((domega_el),(Bh+3*neqel), dU_el, 3, 1, neqel);
		    if(!check) printf( "Problems with matX \n");

/* Calculation of the incremental Yaumann constitutive rate change */

		    *(dstress_el) = *(dstrain_el)*D11 +
			*(dstrain_el+1)*D12+
			*(dstrain_el+2)*D13;
		    *(dstress_el+1) = *(dstrain_el)*D21 +
			*(dstrain_el+1)*D22 +
			*(dstrain_el+2)*D23 ;
		    *(dstress_el+2) = *(dstrain_el)*D31 +
			*(dstrain_el+1)*D32 +
			*(dstrain_el+2)*D33;
		    *(dstress_el+3) = *(dstrain_el+3)*G;
		    *(dstress_el+4) = *(dstrain_el+4)*G;
		    *(dstress_el+5) = *(dstrain_el+5)*G;

/* Update the global stress matrix at half time */
			
		    stress[k].pt[j].xx += *(dstress_el) +
			*(domega_el)*stress[k].pt[j].xy -
			*(domega_el+1)*stress[k].pt[j].zx;
		    stress[k].pt[j].yy +=  *(dstress_el+1) -
			*(domega_el)*stress[k].pt[j].xy +
			*(domega_el+2)*stress[k].pt[j].yz;
		    stress[k].pt[j].zz += *(dstress_el+2) +
			*(domega_el+1)*stress[k].pt[j].zx -
			*(domega_el+2)*stress[k].pt[j].yz;
		    stress[k].pt[j].xy += *(dstress_el+3) +
			.5*(-*(domega_el+1)*stress[k].pt[j].yz +
			*(domega_el+2)*stress[k].pt[j].zx +
			*(domega_el)*(stress[k].pt[j].yy -
			stress[k].pt[j].xx));
		    stress[k].pt[j].zx += *(dstress_el+4) +
			.5*(*(domega_el)*stress[k].pt[j].yz -
			*(domega_el+2)*stress[k].pt[j].xy +
			*(domega_el+1)*(stress[k].pt[j].xx -
			stress[k].pt[j].zz));
		    stress[k].pt[j].yz += *(dstress_el+5) +
			.5*(-*(domega_el)*stress[k].pt[j].zx +
			*(domega_el+1)*stress[k].pt[j].xy +
			*(domega_el+2)*(stress[k].pt[j].zz -
			stress[k].pt[j].yy));

/* Update the global stress matrix at full time */

		    stress[k].pt[j].xx += +
			*(domega_el)*stress[k].pt[j].xy -
			*(domega_el+1)*stress[k].pt[j].zx;
		    stress[k].pt[j].yy += -
			*(domega_el)*stress[k].pt[j].xy +
			*(domega_el+2)*stress[k].pt[j].yz;
		    stress[k].pt[j].zz += +
			*(domega_el+1)*stress[k].pt[j].zx -
			*(domega_el+2)*stress[k].pt[j].yz;
		    stress[k].pt[j].xy += +
			.5*(-*(domega_el+1)*stress[k].pt[j].yz +
			*(domega_el+2)*stress[k].pt[j].zx +
			*(domega_el)*(stress[k].pt[j].yy -
			stress[k].pt[j].xx));
		    stress[k].pt[j].zx += +
			.5*(*(domega_el)*stress[k].pt[j].yz -
			*(domega_el+2)*stress[k].pt[j].xy +
			*(domega_el+1)*(stress[k].pt[j].xx -
			stress[k].pt[j].zz));
		    stress[k].pt[j].yz += +
			.5*(-*(domega_el)*stress[k].pt[j].zx +
			*(domega_el+1)*stress[k].pt[j].xy +
			*(domega_el+2)*(stress[k].pt[j].zz -
			stress[k].pt[j].yy));

		    if(Passemble_flag)
		    {
			memset(P_temp,0,neqel*sof);

/* Calculation of the element stress matrix at full time */

			*(stress_el) = stress[k].pt[j].xx;
			*(stress_el+1) = stress[k].pt[j].yy;
			*(stress_el+2) = stress[k].pt[j].zz;
			*(stress_el+3) = stress[k].pt[j].xy; 
			*(stress_el+4) = stress[k].pt[j].zx;
			*(stress_el+5) = stress[k].pt[j].yz;

			wXdet = *(w+j)*(*(deth+j));

/* Assembly of the element P matrix = (B transpose)*stress_el  */

			check=matXT(P_temp, B, stress_el, neqel, 1, sdim);
			if(!check) printf( "Problems with matXT \n");

			for( i1 = 0; i1 < neqel; ++i1 )
			{
				/*printf("rrrrr %14.9f \n",*(P_temp+i1));*/
				/*printf("rrrrr %14.9f \n",*(P_temp+i1)*wXdet);*/
				*(P_el+i1) += *(P_temp+i1)*wXdet;
			}
		    }
		}

		if(Passemble_flag)
		{

/* Assembly of the global P matrix */

		    for( j = 0; j < neqel; ++j )
		    {
			*(P_global+*(dof_el+j)) += *(P_el+j);
		    }
		}
	}
	return 1;
}

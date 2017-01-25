/*
    This utility function assembles the P_global matrix for a finite
    element program which does analysis on a tetrahedron which can behave
    nonlinearly.  It can also do assembly for the P_global_CG matrix
    if non-linear analysis using the conjugate gradient method is chosen.

	        Updated 11/24/08

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
#include "../tetra/teconst.h"
#include "../tetra/testruct.h"

#define SMALL      1.e-20

extern int numel, numnp, dof, sof;
extern int Passemble_flag, Passemble_CG_flag;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int tetraB(double *,double *);

int dotX(double *, double *, double *, int);

int te2Passemble( int *connect, double *coord, double *coordh, int *el_matl,
	MATL *matl, double *P_global, SDIM *stress, double *dU, double *P_global_CG,
	double *U)
{
	int i, i1, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node;
	int matl_num;
	double Emod, Pois, G;
	double fdum, fdum2, volume_el;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], Bh[soB], stressd[sdim];
	double DB[soB], K_temp[neqlsq], K_el[neqlsq], U_el[neqel];
	double P_el[neqel], P_temp[neqel];
	double dU_el[neqel], domega_el[1], coord_el[npel*nsd], coordh_el[npel*nsd],
		coord_el_trans[npel*nsd], coordh_el_trans[npel*nsd];
	double stress_el[sdim], dstress_el[sdim], dstrain_el[sdim];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double X1h, X2h, X3h, X4h, Y1h, Y2h, Y3h, Y4h, Z1h, Z2h, Z3h, Z4h;
	double a1h, a2h, a3h, a4h, b1h, b2h, b3h, b4h, c1h, c2h, c3h, c4h;
	double det[num_int];
	int i2,i2m1,i2m2;

	memset(P_global,0,dof*sof);
	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;

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

/* Create the dU_el vector for one element */

			*(dU_el+ndof*j) = *(dU+ndof*node);
			*(dU_el+ndof*j+1) = *(dU+ndof*node+1);
			*(dU_el+ndof*j+2) = *(dU+ndof*node+2);

/* Create the coord vector and coordh_trans for one element */

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));
			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

/* Create the coord vector and coordh_trans for one element */

			*(coordh_el+nsd*j)=*(coordh+*(sdof_el+nsd*j));
			*(coordh_el+nsd*j+1)=*(coordh+*(sdof_el+nsd*j+1));
			*(coordh_el+nsd*j+2)=*(coordh+*(sdof_el+nsd*j+2));
			*(coordh_el_trans+j)=*(coordh+*(sdof_el+nsd*j));
			*(coordh_el_trans+npel*1+j)=*(coordh+*(sdof_el+nsd*j+1));
			*(coordh_el_trans+npel*2+j)=*(coordh+*(sdof_el+nsd*j+2));
		}

/* Assembly of the B and Bh matrix.

   This is taken from "Fundamentals of the Finite Element Method" by
   Hartley Grandin Jr., page 383-387. 

   I have changed the node numbering to be consistant with the
   numbering of the brick element, specifically the counter-clockwise
   numbering scheme.  This means that node 4 in Grandin is node 1 in
   SLFFEA, node 1 becomes node 2, node 2 is node 3, and node 3 goes
   to node 4.  So X4 -> X1, X1 -> X2, X2 -> X3, X3 -> X4, and similarly
   for Y and Z.  This also means a4 -> a1, a1 -> a2, a2 -> a3, a3 -> a4
   and similarly for b and c.  Everything else is the same.
*/

/* At full time */

		X1 = *(coord_el_trans);
		X2 = *(coord_el_trans + 1);
		X3 = *(coord_el_trans + 2);
		X4 = *(coord_el_trans + 3);

		Y1 = *(coord_el_trans + npel*1);
		Y2 = *(coord_el_trans + npel*1 + 1);
		Y3 = *(coord_el_trans + npel*1 + 2);
		Y4 = *(coord_el_trans + npel*1 + 3);

		Z1 = *(coord_el_trans + npel*2);
		Z2 = *(coord_el_trans + npel*2 + 1);
		Z3 = *(coord_el_trans + npel*2 + 2);
		Z4 = *(coord_el_trans + npel*2 + 3);

		a2 = (Y3 - Y1)*(Z4 - Z1) - (Y4 - Y1)*(Z3 - Z1);
		a3 = (Y4 - Y1)*(Z2 - Z1) - (Y2 - Y1)*(Z4 - Z1);
		a4 = (Y2 - Y1)*(Z3 - Z1) - (Y3 - Y1)*(Z2 - Z1);
		a1 = -(a2 + a3 + a4);

		b2 = (X4 - X1)*(Z3 - Z1) - (X3 - X1)*(Z4 - Z1);
		b3 = (X2 - X1)*(Z4 - Z1) - (X4 - X1)*(Z2 - Z1);
		b4 = (X3 - X1)*(Z2 - Z1) - (X2 - X1)*(Z3 - Z1);
		b1 = -(b2 + b3 + b4);

		c2 = (X3 - X1)*(Y4 - Y1) - (X4 - X1)*(Y3 - Y1);
		c3 = (X4 - X1)*(Y2 - Y1) - (X2 - X1)*(Y4 - Y1);
		c4 = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);
		c1 = -(c2 + c3 + c4);

		fdum = (X2 - X1)*a2 + (Y2 - Y1)*b2 + (Z2 - Z1)*c2;

		volume_el = pt1667*fdum;

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 6*Vol:%f\n", k, fdum);
			/*return 0;*/
		}

/* At 1/2 time */

		X1h = *(coordh_el_trans);
		X2h = *(coordh_el_trans + 1);
		X3h = *(coordh_el_trans + 2);
		X4h = *(coordh_el_trans + 3);

		Y1h = *(coordh_el_trans + npel*1);
		Y2h = *(coordh_el_trans + npel*1 + 1);
		Y3h = *(coordh_el_trans + npel*1 + 2);
		Y4h = *(coordh_el_trans + npel*1 + 3);

		Z1h = *(coordh_el_trans + npel*2);
		Z2h = *(coordh_el_trans + npel*2 + 1);
		Z3h = *(coordh_el_trans + npel*2 + 2);
		Z4h = *(coordh_el_trans + npel*2 + 3);

		a2h = (Y3h - Y1h)*(Z4h - Z1h) - (Y4h - Y1h)*(Z3h - Z1h);
		a3h = (Y4h - Y1h)*(Z2h - Z1h) - (Y2h - Y1h)*(Z4h - Z1h);
		a4h = (Y2h - Y1h)*(Z3h - Z1h) - (Y3h - Y1h)*(Z2h - Z1h);
		a1h = -(a2h + a3h + a4h);

		b2h = (X4h - X1h)*(Z3h - Z1h) - (X3h - X1h)*(Z4h - Z1h);
		b3h = (X2h - X1h)*(Z4h - Z1h) - (X4h - X1h)*(Z2h - Z1h);
		b4h = (X3h - X1h)*(Z2h - Z1h) - (X2h - X1h)*(Z3h - Z1h);
		b1h = -(b2h + b3h + b4h);

		c2h = (X3h - X1h)*(Y4h - Y1h) - (X4h - X1h)*(Y3h - Y1h);
		c3h = (X4h - X1h)*(Y2h - Y1h) - (X2h - X1h)*(Y4h - Y1h);
		c4h = (X2h - X1h)*(Y3h - Y1h) - (X3h - X1h)*(Y2h - Y1h);
		c1h = -(c2h + c3h + c4h);

		fdum2 = (X2h - X1h)*a2h + (Y2h - Y1h)*b2h + (Z2h - Z1h)*c2h;

		/*for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);
			printf("rr %4d %16.8e %16.8e %16.8e\n",k, *(dU_el+ndof*j), *(dU_el+ndof*j+1), *(dU_el+ndof*j+2));
			printf("rr %4d %16.8e %16.8e %16.8e\n",k, *(dU+ndof*node), *(dU+ndof*node+1), *(dU+ndof*node+2));
		}*/

		memset(P_el,0,neqel*sof);
		if(Passemble_CG_flag)
		{
		    memset(U_el,0,neqel*sof);
		    memset(K_el,0,neqlsq*sof);
		}

/* I could memset "U_el", "K_el", "DB", and "K_temp" together, but they are
   normally broken up by a loop over the integration points, so I will keep
   it seperate to be consistent.
*/

		memset(B,0,soB*sof);
		memset(Bh,0,soB*sof);
		memset(P_temp,0,neqel*sof);

		if(Passemble_CG_flag)
		{
		    memset(DB,0,soB*sof);
		    memset(K_temp,0,neqlsq*sof);
		}

/* Assembly of the B matrix at full time - Assembly of the Bh matrix at 1/2 time */

		*(B)   = a1;               *(Bh)   = a1h;
		*(B+3) = a2;               *(Bh+3) = a2h;
		*(B+6) = a3;               *(Bh+6) = a3h;
		*(B+9) = a4;               *(Bh+9) = a4h;

		*(B+13) = b1;              *(Bh+13) = b1h;
		*(B+16) = b2;              *(Bh+16) = b2h;
		*(B+19) = b3;              *(Bh+19) = b3h;
		*(B+22) = b4;              *(Bh+22) = b4h;

		*(B+26) = c1;              *(Bh+26) = c1h;
		*(B+29) = c2;              *(Bh+29) = c2h;
		*(B+32) = c3;              *(Bh+32) = c3h;
		*(B+35) = c4;              *(Bh+35) = c4h;

		*(B+36) = b1;              *(Bh+36) = b1h;
		*(B+37) = a1;              *(Bh+37) = a1h;
		*(B+39) = b2;              *(Bh+39) = b2h;
		*(B+40) = a2;              *(Bh+40) = a2h;
		*(B+42) = b3;              *(Bh+42) = b3h;
		*(B+43) = a3;              *(Bh+43) = a3h;
		*(B+45) = b4;              *(Bh+45) = b4h;
		*(B+46) = a4;              *(Bh+46) = a4h;

		*(B+48) = c1;              *(Bh+48) = c1h;
		*(B+50) = a1;              *(Bh+50) = a1h;
		*(B+51) = c2;              *(Bh+51) = c2h;
		*(B+53) = a2;              *(Bh+53) = a2h;
		*(B+54) = c3;              *(Bh+54) = c3h;
		*(B+56) = a3;              *(Bh+56) = a3h;
		*(B+57) = c4;              *(Bh+57) = c4h;
		*(B+59) = a4;              *(Bh+59) = a4h;

		*(B+61) = c1;              *(Bh+61) = c1h;
		*(B+62) = b1;              *(Bh+62) = b1h;
		*(B+64) = c2;              *(Bh+64) = c2h;
		*(B+65) = b2;              *(Bh+65) = b2h;
		*(B+67) = c3;              *(Bh+67) = c3h;
		*(B+68) = b3;              *(Bh+68) = b3h;
		*(B+70) = c4;              *(Bh+70) = c4h;
		*(B+71) = b4;              *(Bh+71) = b4h;

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
		    
		stress[k].xx += *(dstress_el) +
		    *(domega_el)*stress[k].xy -
		    *(domega_el+1)*stress[k].zx;
		stress[k].yy +=  *(dstress_el+1) -
		    *(domega_el)*stress[k].xy +
		    *(domega_el+2)*stress[k].yz;
		stress[k].zz += *(dstress_el+2) +
		    *(domega_el+1)*stress[k].zx -
		    *(domega_el+2)*stress[k].yz;
		stress[k].xy += *(dstress_el+3) +
		    .5*(-*(domega_el+1)*stress[k].yz +
		    *(domega_el+2)*stress[k].zx +
		    *(domega_el)*(stress[k].yy -
		    stress[k].xx));
		stress[k].zx += *(dstress_el+4) +
		    .5*(*(domega_el)*stress[k].yz -
		    *(domega_el+2)*stress[k].xy +
		    *(domega_el+1)*(stress[k].xx -
		    stress[k].zz));
		stress[k].yz += *(dstress_el+5) +
		    .5*(-*(domega_el)*stress[k].zx +
		    *(domega_el+1)*stress[k].xy +
		    *(domega_el+2)*(stress[k].zz -
		    stress[k].yy));

/* Update the global stress matrix at full time */

		stress[k].xx += +
		    *(domega_el)*stress[k].xy -
		    *(domega_el+1)*stress[k].zx;
		stress[k].yy += -
		    *(domega_el)*stress[k].xy +
		    *(domega_el+2)*stress[k].yz;
		stress[k].zz += +
		    *(domega_el+1)*stress[k].zx -
		    *(domega_el+2)*stress[k].yz;
		stress[k].xy += +
		    .5*(-*(domega_el+1)*stress[k].yz +
		    *(domega_el+2)*stress[k].zx +
		    *(domega_el)*(stress[k].yy -
		    stress[k].xx));
		stress[k].zx += +
		    .5*(*(domega_el)*stress[k].yz -
		    *(domega_el+2)*stress[k].xy +
		    *(domega_el+1)*(stress[k].xx -
		    stress[k].zz));
		stress[k].yz += +
		    .5*(-*(domega_el)*stress[k].zx +
		    *(domega_el+1)*stress[k].xy +
		    *(domega_el+2)*(stress[k].zz -
		    stress[k].yy));

		if(Passemble_flag)
		{

/* Calculation of the element stress matrix at full time */

		    *(stress_el) = stress[k].xx;
		    *(stress_el+1) = stress[k].yy;
		    *(stress_el+2) = stress[k].zz;
		    *(stress_el+3) = stress[k].xy; 
		    *(stress_el+4) = stress[k].zx;
		    *(stress_el+5) = stress[k].yz;

/* Assembly of the element P matrix = (B transpose)*stress_el  */

		    check=matXT(P_temp, B, stress_el, neqel, 1, sdim);
		    if(!check) printf( "Problems with matXT \n");

		    for( i1 = 0; i1 < neqel; ++i1 )
		    {
			   /*printf("rrrrr %14.9f \n",*(P_temp+i1));*/
			   /*printf("rrrrr %4d %14.9f \n",k, *(P_temp+i1)/(36.0*volume_el));*/
			   *(P_el+i1) += *(P_temp+i1)/(36.0*volume_el);
		    }
		}

		if(Passemble_CG_flag)
		{
/*
    Note that I do not use Bh for DB when the conjugate gradient
    method is used.  This is because I want to keep things consistent with
    teConjPassemble.
*/
		    for( i1 = 0; i1 < neqel; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel*1+i1)*D12+
				*(B+neqel*2+i1)*D13;
			*(DB+neqel*1+i1) = *(B+i1)*D21+
				*(B+neqel*1+i1)*D22+
				*(B+neqel*2+i1)*D23;
			*(DB+neqel*2+i1) = *(B+i1)*D31+
				*(B+neqel*1+i1)*D32+
				*(B+neqel*2+i1)*D33;
			*(DB+neqel*3+i1) = *(B+neqel*3+i1)*G;
			*(DB+neqel*4+i1) = *(B+neqel*4+i1)*G;
			*(DB+neqel*5+i1) = *(B+neqel*5+i1)*G;
		    }

		    check=matXT(K_el, B, DB, neqel, neqel, sdim);
		    if(!check) printf( "Problems with matXT \n");

		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(K_el+i2) /= 36.0*volume_el;
		    }
		}

		if(Passemble_flag)
		{

/* Assembly of the global P_global matrix */

		    for( j = 0; j < neqel; ++j )
		    {
			*(P_global+*(dof_el+j)) += *(P_el+j);
		    }
		}
		if(Passemble_CG_flag)
		{

/* Assembly of the global conjugate gradient P_global_CG matrix */

		    for( j = 0; j < neqel; ++j )
		    {
			*(U_el + j) = *(U + *(dof_el+j));
		    }

		    check = matX(P_el, K_el, U_el, neqel, 1, neqel);
		    if(!check) printf( "Problems with matX \n");

		    for( j = 0; j < neqel; ++j )
		    {
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		    }
		}
	}

	return 1;
}



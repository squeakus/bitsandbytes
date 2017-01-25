/*
    This utility function assembles the Mass and force matrix for a finite
    element program which does analysis on a tetrahedron which can behave
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

    Note that I do not use Bh for DB in te2Passemble when the conjugate
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
#include "../tetra/teconst.h"
#include "../tetra/testruct.h"

#define  DEBUG         0

extern int numel, numnp, dof, sof;
extern double shg[sosh], shg_node[sosh], shl[sosh], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Vol0;

int cubic( double *);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int tetraB( double *,double *);

int teshg( double *, int, double *, double *, double *);

int teFMassemble( int *connect, double *coord, double *coordh, int *el_matl,
	double *force, double *mass, MATL *matl, double *U, double *Voln)
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node;
	int matl_num;
	double Emod, Pois, G;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], Bh[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double force_el[neqel], U_el[neqel];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd];
	double coordh_el[npel*nsd], coordh_el_trans[npel*nsd];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double X1h, X2h, X3h, X4h, Y1h, Y2h, Y3h, Y4h, Z1h, Z2h, Z3h, Z4h;
	double a1h, a2h, a3h, a4h, b1h, b2h, b3h, b4h, c1h, c2h, c3h, c4h;
	double stress_el[sdim], strain_el[sdim], invariant[nsd],
		yzsq, zxsq, xysq, xxyy;
	double det[1], wXdet;
	double fdum, fdum2;
	double mass_el[neqel];

/* initialize all variables  */
	memset(mass,0,dof*sof);

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

/* Assembly of the B matrix.

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

#if !DEBUG
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

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 6*Vol:%f\n", k, fdum);
			/*return 0;*/
		}

		*(Voln + k) = pt1667*fdum;

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

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);
		memset(force_el,0,neqel*sof);

		memset(B,0,soB*sof);
		memset(Bh,0,soB*sof);
		memset(DB,0,soB*sof);

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

#endif

#if DEBUG

/* The code below is for debugging.  It uses shape functions
   to calculate the B matrix.  Normally, we loop through the number
   of integration points (num_int), but for tetrahedrons, the shape
   function derivatives are constant.
*/
		check=teshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with teshg \n");

		check = tetraB(shg, B);
		if(!check) printf( "Problems with tetraB \n");

		check=teshg(deth, k, shl, shgh, coordh_el_trans);
		if(!check) printf( "Problems with teshg \n");

		check = tetraB(shgh, Bh);
		if(!check) printf( "Problems with tetraB \n");
#endif

		/*for( i1 = 0; i1 < soB; ++i1 )
		{
			*(B+i1) *= -1.0;
		}*/

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

		check=matXT(K_el, B, DB, neqel, neqel, sdim);
		if(!check) printf( "Problems with matXT \n");

#if !DEBUG
		for( i2 = 0; i2 < neqlsq; ++i2 )
		{
		    *(K_el+i2) /= 36.0*(*(Voln+k));
		}
#endif

#if DEBUG
/* The code below is for debugging the code.  Normally, I would use
   wXdet rather than pt1667*(*(det)), but we are really using the one 
   point rule, since the derivative of the shape functions are
   constant, and w = 1.0.  Also note that the determinant does not change
   because the derivative of the shape functions is constant so we
   only have det[1].

   A factor of 1/6 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		for( j = 0; j < neqlsq; ++j )
		{
			*(K_el + j) *= pt1667*(*(det));
		}
#endif

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(force_el, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

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


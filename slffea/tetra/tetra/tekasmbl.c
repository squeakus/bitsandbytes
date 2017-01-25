/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a tetrahedral element.

		Updated 11/22/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "teconst.h"
#include "testruct.h"

#define  DEBUG         0

extern int analysis_flag, dof, neqn, numel, numnp, sof;
extern int LU_decomp_flag, numel_K, numel_P, numel_surf;
extern double shg[sosh], shg_node[sosh], shl[sosh], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Vol0;

int cubic( double *);

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int tetraB( double *,double *);

int teshg( double *, int, double *, double *, double *);

int teKassemble(double *A, int *connect, int *connect_surf, double *coord, int *el_matl,
	int *el_matl_surf, double *force, int *id, int *idiag, double *K_diag, int *lm,
	MATL *matl, double *node_counter, SDIM *strain, SDIM *strain_node,
	SDIM *stress, SDIM *stress_node, double *U, double *Voln)
{
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, surf_el_counter, surface_el_flag;
	int matl_num;
	double Emod, Pois, G;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double force_el[neqel], U_el[neqel];
	double coord_el_trans[npel*nsd];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double stress_el[sdim], strain_el[sdim], invariant[nsd],
		yzsq, zxsq, xysq, xxyy;
	double det[1], wXdet;
	double fdum;

	surf_el_counter = 0;
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

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;

/* Count the number of times a particular node is part of an element */

			if(analysis_flag == 1)
				*(node_counter + node) += 1.0;
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

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(force_el,0,neqel*sof);

		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);

		*(B) = a1;
		*(B+3) = a2;
		*(B+6) = a3;
		*(B+9) = a4;

		*(B+13) = b1;
		*(B+16) = b2;
		*(B+19) = b3;
		*(B+22) = b4;

		*(B+26) = c1;
		*(B+29) = c2;
		*(B+32) = c3;
		*(B+35) = c4;

		*(B+36) = b1;
		*(B+37) = a1;
		*(B+39) = b2;
		*(B+40) = a2;
		*(B+42) = b3;
		*(B+43) = a3;
		*(B+45) = b4;
		*(B+46) = a4;

		*(B+48) = c1;
		*(B+50) = a1;
		*(B+51) = c2;
		*(B+53) = a2;
		*(B+54) = c3;
		*(B+56) = a3;
		*(B+57) = c4;
		*(B+59) = a4;

		*(B+61) = c1;
		*(B+62) = b1;
		*(B+64) = c2;
		*(B+65) = b2;
		*(B+67) = c3;
		*(B+68) = b3;
		*(B+70) = c4;
		*(B+71) = b4;
#endif

#if DEBUG

/* The code below is for debugging.  It uses shape functions
   to calculate the B matrix.  Normally, we loop through the number
   of integration points (num_int), but for tetrahedrons, the shape
   function derivatives are constant.
*/
		check=teshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with teshg \n");

		check = tetraB(shg,B);
		if(!check) printf( "Problems with tetraB \n");
#endif

		/*for( i1 = 0; i1 < soB; ++i1 )
		{
			*(B+i1) *= -1.0;
		}*/

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

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel; ++j )
			{
				*(force + *(dof_el+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(A, idiag, K_el, (lm + k*neqel),
				neqel);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(A, dof_el, k, K_diag, K_el,
				neqel, neqlsq, numel_K);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel; ++j )
			{
				*(force + *(dof_el+j)) += *(force_el + j);
			}

/* Calculate the element stresses */

			surface_el_flag = 0;
			for( j = 0; j < npel; ++j )
			{

/* Determine which elements are on the surface */

			   node = *(connect+npel*k+j);

			   if( *(node_counter + node) < 7.5 )
				surface_el_flag = 1;

/* Calculation of the local strain matrix */

			}

			memset(stress_el,0,sdim*sof);
			memset(strain_el,0,sdim*sof);

			check=matX(strain_el, B, U_el, sdim, 1, neqel );
			if(!check) printf( "Problems with matX \n");

#if 0
			for( i1 = 0; i1 < sdim; ++i1 )
			{
				printf("%12.8f ",*(stress_el+1));
				/*printf("%12.2f ",*(stress_el+2));
				printf("%12.8f ",*(B+i1));*/
			}
			printf("\n");
#endif

#if !DEBUG
			*(strain_el) /= 6.0*(*(Voln + k));
			*(strain_el + 1) /= 6.0*(*(Voln + k));
			*(strain_el + 2) /= 6.0*(*(Voln + k));
			*(strain_el + 3) /= 6.0*(*(Voln + k));
			*(strain_el + 4) /= 6.0*(*(Voln + k));
			*(strain_el + 5) /= 6.0*(*(Voln + k));
#endif

/* Update of the global strain matrix */

			strain[k].xx = *(strain_el);
			strain[k].yy = *(strain_el+1);
			strain[k].zz = *(strain_el+2);
			strain[k].xy = *(strain_el+3);
			strain[k].zx = *(strain_el+4);
			strain[k].yz = *(strain_el+5);

/* Calculate the principal strians */

			memset(invariant,0,nsd*sof);
			xysq = strain[k].xy*strain[k].xy;
			zxsq = strain[k].zx*strain[k].zx;
			yzsq = strain[k].yz*strain[k].yz;
			xxyy = strain[k].xx*strain[k].yy;

			*(invariant) = - strain[k].xx -
				strain[k].yy - strain[k].zz;
			*(invariant+1) = xxyy +
				strain[k].yy*strain[k].zz +
				strain[k].zz*strain[k].xx -
				yzsq - zxsq - xysq;
			*(invariant+2) = - xxyy*strain[k].zz -
				2*strain[k].yz*strain[k].zx*strain[k].xy +
				yzsq*strain[k].xx + zxsq*strain[k].yy +
				xysq*strain[k].zz;
				
			check = cubic(invariant);

			strain[k].I = *(invariant);
			strain[k].II = *(invariant+1);
			strain[k].III = *(invariant+2);

/* Calculation of the local stress matrix */

			*(stress_el) = strain[k].xx*D11+
				strain[k].yy*D12+
				strain[k].zz*D13;
			*(stress_el+1) = strain[k].xx*D21+
				strain[k].yy*D22+
				strain[k].zz*D23;
			*(stress_el+2) = strain[k].xx*D31+
				strain[k].yy*D32+
				strain[k].zz*D33;
			*(stress_el+3) = strain[k].xy*G;
			*(stress_el+4) = strain[k].zx*G;
			*(stress_el+5) = strain[k].yz*G; 

/* Update of the global stress matrix */

			stress[k].xx += *(stress_el);
			stress[k].yy += *(stress_el+1);
			stress[k].zz += *(stress_el+2);
			stress[k].xy += *(stress_el+3);
			stress[k].zx += *(stress_el+4);
			stress[k].yz += *(stress_el+5);

/* Calculate the principal stresses */

			memset(invariant,0,nsd*sof);
			xysq = stress[k].xy*stress[k].xy;
			zxsq = stress[k].zx*stress[k].zx;
			yzsq = stress[k].yz*stress[k].yz;
			xxyy = stress[k].xx*stress[k].yy;

			*(invariant) = - stress[k].xx -
				stress[k].yy - stress[k].zz;
			*(invariant+1) = xxyy +
				stress[k].yy*stress[k].zz +
				stress[k].zz*stress[k].xx -
				yzsq - zxsq - xysq;
			*(invariant+2) = - xxyy*stress[k].zz -
				2*stress[k].yz*stress[k].zx*stress[k].xy +
				yzsq*stress[k].xx + zxsq*stress[k].yy +
				xysq*stress[k].zz;
				
			check = cubic(invariant);

			stress[k].I = *(invariant);
			stress[k].II = *(invariant+1);
			stress[k].III = *(invariant+2);

			for( j = 0; j < npel; ++j )
			{
			    node = *(connect+npel*k+j);

/* Add all the strains for a particular node from all the elements which share that node */

			    strain_node[node].xx += strain[k].xx;
			    strain_node[node].yy += strain[k].yy;
			    strain_node[node].zz += strain[k].zz;
			    strain_node[node].xy += strain[k].xy;
			    strain_node[node].zx += strain[k].zx;
			    strain_node[node].yz += strain[k].yz;
			    strain_node[node].I += strain[k].I;
			    strain_node[node].II += strain[k].II;
			    strain_node[node].III += strain[k].III;

/* Add all the stresses for a particular node from all the elements which share that node */

			    stress_node[node].xx += stress[k].xx;
			    stress_node[node].yy += stress[k].yy;
			    stress_node[node].zz += stress[k].zz;
			    stress_node[node].xy += stress[k].xy;
			    stress_node[node].zx += stress[k].zx;
			    stress_node[node].yz += stress[k].yz;
			    stress_node[node].I += stress[k].I;
			    stress_node[node].II += stress[k].II;
			    stress_node[node].III += stress[k].III;
			}

/*
			printf("%14.6e ",stress[k].xx);
			printf("%14.6e ",stress[k].yy);
			printf("%14.6e ",stress[k].zz);
			printf("%14.6e ",stress[k].yz);
			printf("%14.6e ",stress[k].zx);
			printf("%14.6e ",stress[k].xy);
			printf( "\n");
*/
			/*printf( "\n");*/

/* Create surface connectivity elements */

			if(surface_el_flag)
			{
			    for( j = 0; j < npel; ++j)
			    {
				*(connect_surf + npel*surf_el_counter + j) =
					*(connect + npel*k + j);
			    }
			    *(el_matl_surf + surf_el_counter) = matl_num;
			    ++surf_el_counter;
			}
		}
		numel_surf = surf_el_counter;
	}

	if(analysis_flag == 1)
	{

/* Contract the global force matrix using the id array only if LU decomposition
   is used. */

	  if(LU_decomp_flag)
	  {
	     counter = 0;
	     for( i = 0; i < dof ; ++i )
	     {
		if( *(id + i ) > -1 )
		{
			*(force + counter ) = *(force + i );
			++counter;
		}
	     }
	  }
	}
	if(analysis_flag == 2)
	{

/* Average all the stresses and strains at the nodes */

	  for( i = 0; i < numnp ; ++i )
	  {
		   strain_node[i].xx /= *(node_counter + i);
		   strain_node[i].yy /= *(node_counter + i);
		   strain_node[i].zz /= *(node_counter + i);
		   strain_node[i].xy /= *(node_counter + i);
		   strain_node[i].zx /= *(node_counter + i);
		   strain_node[i].yz /= *(node_counter + i);
		   strain_node[i].I /= *(node_counter + i);
		   strain_node[i].II /= *(node_counter + i);
		   strain_node[i].III /= *(node_counter + i);

		   stress_node[i].xx /= *(node_counter + i);
		   stress_node[i].yy /= *(node_counter + i);
		   stress_node[i].zz /= *(node_counter + i);
		   stress_node[i].xy /= *(node_counter + i);
		   stress_node[i].zx /= *(node_counter + i);
		   stress_node[i].yz /= *(node_counter + i);
		   stress_node[i].I /= *(node_counter + i);
		   stress_node[i].II /= *(node_counter + i);
		   stress_node[i].III /= *(node_counter + i);
	  }
	}

	return 1;
}


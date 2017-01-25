/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a brick.

		Updated 9/26/01

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
#include "brconst.h"
#include "brstruct.h"

extern int analysis_flag, dof, neqn, numel, numnp, sof;
extern int gauss_stress_flag;
extern int LU_decomp_flag, numel_K, numel_P, numel_surf;
extern double shg[sosh], shg_node[sosh], shl[sosh], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Vol0;

int cubic( double *);

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int brickB( double *,double *);

int brshg( double *, int, double *, double *, double *);

int brstress_shg( double *, int, double *, double *, double * );
	       
int brKassemble(double *A, int *connect, int *connect_surf, double *coord, int *el_matl,
	int *el_matl_surf, double *force, int *id, int *idiag, double *K_diag, int *lm,
	MATL *matl, double *node_counter, STRAIN *strain, SDIM *strain_node,
	STRESS *stress, SDIM *stress_node, double *U)
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
	double stress_el[sdim], strain_el[sdim], invariant[nsd],
		yzsq, zxsq, xysq, xxyy;
	double det[num_int], wXdet;

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


/* Assembly of the shg matrix for each integration point */

		check=brshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* The loop over j below calculates the 8 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(force_el,0,neqel*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_temp,0,neqlsq*sof);

/* Assembly of the B matrix */

		    check = brickB((shg+npel*(nsd+1)*j),B);
		    if(!check) printf( "Problems with brickB \n");

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

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_temp, B, DB, neqel, neqel, sdim);
		    if(!check) printf( "Problems with matXT \n");
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(K_el+i2) += *(K_temp+i2)*wXdet;
		    }
		}

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

/* Assembly of the brstress_shg matrix for each nodal point */

/*
   brstress_shg calculates shg at the nodes.  It is more efficient than brshg
   because it removes all the zero multiplications from the calculation of shg. 
   You can use either function when calculating shg at the nodes. 

			    check=brstress_shg(det, k, shl_node2, shg_node, coord_el_trans);
			    check=brshg(det, k, shl_node, shg_node, coord_el_trans);
*/

			if(gauss_stress_flag)
			{
/* Calculate shg at integration point */
			    check=brshg(det, k, shl, shg, coord_el_trans);
			    if(!check) printf( "Problems with brshg \n");
			}
			else
			{
/* Calculate shg at nodal point */
			    check=brstress_shg(det, k, shl_node2, shg_node, coord_el_trans);
			    if(!check) printf( "Problems with brstress \n");
			}

			surface_el_flag = 0;
			for( j = 0; j < num_int; ++j )
			{

			   memset(B,0,soB*sof);
			   memset(stress_el,0,sdim*sof);
			   memset(strain_el,0,sdim*sof);

/* Determine which elements are on the surface */

			   node = *(connect+npel*k+j);

			   if( *(node_counter + node) < 7.5 )
				surface_el_flag = 1;

/* Assembly of the B matrix */

			   if(gauss_stress_flag)
			   {
/* Calculate B matrix at integration point */
				check = brickB((shg+npel*(nsd+1)*j),B);
				if(!check) printf( "Problems with brickB \n");
			   }
			   else
			   {
/* Calculate B matrix at nodal point */
				check = brickB((shg_node+npel*(nsd+1)*j),B);
				if(!check) printf( "Problems with brickB \n");
			   }
		
/* Calculation of the local strain matrix */

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

/* Update of the global strain matrix */

			   strain[k].pt[j].xx = *(strain_el);
			   strain[k].pt[j].yy = *(strain_el+1);
			   strain[k].pt[j].zz = *(strain_el+2);
			   strain[k].pt[j].xy = *(strain_el+3);
			   strain[k].pt[j].zx = *(strain_el+4);
			   strain[k].pt[j].yz = *(strain_el+5);

/* Calculate the principal strians */

			   memset(invariant,0,nsd*sof);
			   xysq = strain[k].pt[j].xy*strain[k].pt[j].xy;
			   zxsq = strain[k].pt[j].zx*strain[k].pt[j].zx;
			   yzsq = strain[k].pt[j].yz*strain[k].pt[j].yz;
			   xxyy = strain[k].pt[j].xx*strain[k].pt[j].yy;

			   *(invariant) = - strain[k].pt[j].xx -
				strain[k].pt[j].yy - strain[k].pt[j].zz;
			   *(invariant+1) = xxyy +
				strain[k].pt[j].yy*strain[k].pt[j].zz +
				strain[k].pt[j].zz*strain[k].pt[j].xx -
				yzsq - zxsq - xysq;
			   *(invariant+2) = - xxyy*strain[k].pt[j].zz -
				2*strain[k].pt[j].yz*strain[k].pt[j].zx*strain[k].pt[j].xy +
				yzsq*strain[k].pt[j].xx + zxsq*strain[k].pt[j].yy +
				xysq*strain[k].pt[j].zz;
				
			   check = cubic(invariant);

			   strain[k].pt[j].I = *(invariant);
			   strain[k].pt[j].II = *(invariant+1);
			   strain[k].pt[j].III = *(invariant+2);

/* Add all the strains for a particular node from all the elements which share that node */

			   strain_node[node].xx += strain[k].pt[j].xx;
			   strain_node[node].yy += strain[k].pt[j].yy;
			   strain_node[node].zz += strain[k].pt[j].zz;
			   strain_node[node].xy += strain[k].pt[j].xy;
			   strain_node[node].zx += strain[k].pt[j].zx;
			   strain_node[node].yz += strain[k].pt[j].yz;
			   strain_node[node].I += strain[k].pt[j].I;
			   strain_node[node].II += strain[k].pt[j].II;
			   strain_node[node].III += strain[k].pt[j].III;

/* Calculation of the local stress matrix */

			   *(stress_el) = strain[k].pt[j].xx*D11+
				strain[k].pt[j].yy*D12+
				strain[k].pt[j].zz*D13;
			   *(stress_el+1) = strain[k].pt[j].xx*D21+
				strain[k].pt[j].yy*D22+
				strain[k].pt[j].zz*D23;
			   *(stress_el+2) = strain[k].pt[j].xx*D31+
				strain[k].pt[j].yy*D32+
				strain[k].pt[j].zz*D33;
			   *(stress_el+3) = strain[k].pt[j].xy*G;
			   *(stress_el+4) = strain[k].pt[j].zx*G;
			   *(stress_el+5) = strain[k].pt[j].yz*G; 

/* Update of the global stress matrix */

			   stress[k].pt[j].xx += *(stress_el);
			   stress[k].pt[j].yy += *(stress_el+1);
			   stress[k].pt[j].zz += *(stress_el+2);
			   stress[k].pt[j].xy += *(stress_el+3);
			   stress[k].pt[j].zx += *(stress_el+4);
			   stress[k].pt[j].yz += *(stress_el+5);

/* Calculate the principal stresses */

			   memset(invariant,0,nsd*sof);
			   xysq = stress[k].pt[j].xy*stress[k].pt[j].xy;
			   zxsq = stress[k].pt[j].zx*stress[k].pt[j].zx;
			   yzsq = stress[k].pt[j].yz*stress[k].pt[j].yz;
			   xxyy = stress[k].pt[j].xx*stress[k].pt[j].yy;

			   *(invariant) = - stress[k].pt[j].xx -
				stress[k].pt[j].yy - stress[k].pt[j].zz;
			   *(invariant+1) = xxyy +
				stress[k].pt[j].yy*stress[k].pt[j].zz +
				stress[k].pt[j].zz*stress[k].pt[j].xx -
				yzsq - zxsq - xysq;
			   *(invariant+2) = - xxyy*stress[k].pt[j].zz -
				2*stress[k].pt[j].yz*stress[k].pt[j].zx*stress[k].pt[j].xy +
				yzsq*stress[k].pt[j].xx + zxsq*stress[k].pt[j].yy +
				xysq*stress[k].pt[j].zz;
				
			   check = cubic(invariant);

			   stress[k].pt[j].I = *(invariant);
			   stress[k].pt[j].II = *(invariant+1);
			   stress[k].pt[j].III = *(invariant+2);

/* Add all the stresses for a particular node from all the elements which share that node */

			   stress_node[node].xx += stress[k].pt[j].xx;
			   stress_node[node].yy += stress[k].pt[j].yy;
			   stress_node[node].zz += stress[k].pt[j].zz;
			   stress_node[node].xy += stress[k].pt[j].xy;
			   stress_node[node].zx += stress[k].pt[j].zx;
			   stress_node[node].yz += stress[k].pt[j].yz;
			   stress_node[node].I += stress[k].pt[j].I;
			   stress_node[node].II += stress[k].pt[j].II;
			   stress_node[node].III += stress[k].pt[j].III;

/*
			   printf("%14.6e ",stress[k].pt[j].xx);
			   printf("%14.6e ",stress[k].pt[j].yy);
			   printf("%14.6e ",stress[k].pt[j].zz);
			   printf("%14.6e ",stress[k].pt[j].yz);
			   printf("%14.6e ",stress[k].pt[j].zx);
			   printf("%14.6e ",stress[k].pt[j].xy);
			   printf( "\n");
*/
			}
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


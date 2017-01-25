/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a triangle.

		Updated 10/20/06

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
#include "trconst.h"
#include "trstruct.h"

#define  DEBUG         0

extern int analysis_flag, dof, neqn, numel, numnp, plane_stress_flag, sof, flag_3D;
extern int gauss_stress_flag;
extern int LU_decomp_flag, numel_K, numel_P;
extern double shg[sosh], shg_node[sosh], shl[sosh], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Area0;

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int triangleB( double *,double *);

int trshg( double *, int, int, double *, double *, double *);

int trKassemble(double *A, int *connect, double *coord, int *el_matl, double *force,
	int *id, int *idiag, double *K_diag, int *lm, double *local_xyz, MATL *matl,
	double *node_counter, SDIM *strain, SDIM *strain_node, SDIM *stress,
	SDIM *stress_node, double *U, double *Arean)
{
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq36], rotate[nsd2*nsd];
	double force_el[neqel], U_el[neqel], U_el_local[npel*(ndof-1)];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double X1, X2, X3, Y1, Y2, Y3;
	double stress_el[sdim], strain_el[sdim], xxaddyy, xxsubyy, xysq;
	double det[1], wXdet;
	double fdum;

	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* Recalculate lamda for plane stress */

		if(plane_stress_flag)
			lamda = Emod*Pois/(1.0-Pois*Pois);

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Normally, with plane elements, we assume a unit thickness in the transverse direction.  But
   because these elements can be 3 dimensional, I multiply the material property matrix by the
   thickness.  This is justified by equation 5.141a in "Theory of Matrix Structural
   Analysis" by J. S. Przemieniecki on page 86.
*/

		D11 = thickness*(lamda+2.0*mu);
		D12 = thickness*lamda;
		D21 = thickness*lamda;
		D22 = thickness*(lamda+2.0*mu);

		G = mu;
		Gt = thickness*mu;

/* Create the coord_el vector for one element */

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

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

		memset(rotate,0,nsd2*nsd*sof);
		
		if(!flag_3D)
		{
/* For 2-D meshes */
		    X1 = *(coord_el_trans);
		    X2 = *(coord_el_trans + 1);
		    X3 = *(coord_el_trans + 2);

		    Y1 = *(coord_el_trans + npel*1);
		    Y2 = *(coord_el_trans + npel*1 + 1);
		    Y3 = *(coord_el_trans + npel*1 + 2);
		}
		else
		{
/* For 3-D meshes */

/* For 3-D triangle meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  The local basis used for the
   rotation has already been calculated and stored in local_xyz[].  Below, it is
   copied to rotate[].
*/
		    *(rotate)     = *(local_xyz + nsdsq*k);
		    *(rotate + 1) = *(local_xyz + nsdsq*k + 1);
		    *(rotate + 2) = *(local_xyz + nsdsq*k + 2);
		    *(rotate + 3) = *(local_xyz + nsdsq*k + 3);
		    *(rotate + 4) = *(local_xyz + nsdsq*k + 4);
		    *(rotate + 5) = *(local_xyz + nsdsq*k + 5);

/* Put coord_el into local coordinates */

		    dum = nsd*npel;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");

/* Assembly of the B matrix.  This is taken from "Fundamentals of the Finite
   Element Method" by Hartley Grandin Jr., page 201-205.

     It should be noted that I have permutated the node sequences.
     (see Hughes Figure 3.I.5 on page 167; node 3 in Hughes is node 1
     in SLFFEA, node 2 is node 3, and node 1 goes to node 2.)
     This is because I want to be consistant with the tetrahedron.  You can
     read more about this change in teshl for the tetrahedron.

     This change only effects the calculation of the area.  The [B] matrix
     is still the same.
*/

		    X1 = *(coord_el_local);     Y1 = *(coord_el_local + 1);
		    X2 = *(coord_el_local + 2); Y2 = *(coord_el_local + 3);
		    X3 = *(coord_el_local + 4); Y3 = *(coord_el_local + 5);
		}

		*(coord_el_local_trans) = X1;     *(coord_el_local_trans + 3) = Y1;
		*(coord_el_local_trans + 1) = X2; *(coord_el_local_trans + 4) = Y2;
		*(coord_el_local_trans + 2) = X3; *(coord_el_local_trans + 5) = Y3;

/* Area is simply the the cross product of the sides connecting node 1 to node 0
   and node 2 to node 0, divided by 2.
*/

		fdum = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 2*Area:%f\n", k, fdum);
			return 0;
		}

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/

		*(Arean + k) = .5*fdum;

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(K_temp,0,neqlsq*sof);
		memset(force_el,0,neqel*sof);

		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);
		memset(K_local,0,neqlsq36*sof);
#if !DEBUG

/* For [B] below, see "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204.  Despite the permutation of nodes mentioned above, 
   B remains the same.  */

		*(B) = Y2 - Y3;
		*(B+2) = Y3 - Y1;
		*(B+4) = Y1 - Y2;
		*(B+7) = X3 - X2;
		*(B+9) = X1 - X3;
		*(B+11) = X2 - X1;
		*(B+12) = X3 - X2;
		*(B+13) = Y2 - Y3;
		*(B+14) = X1 - X3;
		*(B+15) = Y3 - Y1;
		*(B+16) = X2 - X1;
		*(B+17) = Y1 - Y2;
#endif

#if DEBUG

/* The code below is for debugging the code.  It uses shape functions
   to calculate the B matrix.  Normally, we loop through the number
   of integration points (num_int), but for triangles, the shape
   function derivatives are constant.
*/
		check=trshg(det, k, 2, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with trshg \n");

		check = triangleB(shg,B);
		if(!check) printf( "Problems with triangleB \n");
#endif

		for( i1 = 0; i1 < neqel6; ++i1 )
		{
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel6*1+i1)*D12;
			*(DB+neqel6*1+i1) = *(B+i1)*D21+
				*(B+neqel6*1+i1)*D22;
			*(DB+neqel6*2+i1) = *(B+neqel6*2+i1)*Gt;
		}

		check=matXT(K_local, B, DB, neqel6, neqel6, sdim);
		if(!check) printf( "Problems with matXT  \n");

#if !DEBUG
		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_el + j) = *(K_local + j)/( 4.0*(*(Arean + k)) );
		}
#endif

#if DEBUG
/* The code below is for debugging the code.  Normally, I would use
   wXdet rather than .5*(*(det)), but we are really using the one
   point rule, since the derivative of the shape functions is
   constant, and w = 1.0.  This also means that the determinant does
   not change and we only need *(det+0).

   A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_el + j) = *(K_local + j)*.5*(*(det));
		}
#endif

		if(!flag_3D)
		{
/* For 2-D meshes */
		   for( i = 0; i < npel; ++i )
		   {
			for( j = 0; j < npel; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof*neqel*i + ndof*j) =
				*(K_el + ndof2*neqel6*i + ndof2*j);
			   *(K_temp + ndof*neqel*i + ndof*j + 1) =
				*(K_el + ndof2*neqel6*i + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof*neqel*i + neqel + ndof*j) =
				*(K_el + ndof2*neqel6*i + neqel6 + ndof2*j);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(K_el + ndof2*neqel6*i + neqel6 + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 2) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j) = 0.0;
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 1) = 0.0;
			   *(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 2) = 0.0;
			}
		   }
		   memcpy(K_el, K_temp, neqlsq*sizeof(double));
		}
		else
		{
/* For 3-D meshes */

/* Put K back to global coordinates */

		   check = matXrot2(K_temp, K_el, rotate, neqel6, neqel);
		   if(!check) printf( "Problems with matXrot2 \n");

		   check = rotTXmat2(K_el, rotate, K_temp, neqel, neqel);
		   if(!check) printf( "Problems with rotTXmat2 \n");
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

/* Calculate the element stresses.  Note that strain and stress are
   constant over a 3 node triangle element */

			memset(stress_el,0,sdim*sof);
			memset(strain_el,0,sdim*sof);
			memset(U_el_local,0,neqel6*sof);

/* Calculation of the local strain matrix */

/* Determine local U coordinates */

			if(!flag_3D)
			{
/* For 2-D meshes */
				*(U_el_local) = *(U_el);
				*(U_el_local + 1) = *(U_el + 1);

				*(U_el_local + 2) = *(U_el + 3);
				*(U_el_local + 3) = *(U_el + 4);

				*(U_el_local + 4) = *(U_el + 6);
				*(U_el_local + 5) = *(U_el + 7);
			}
			else
			{
/* For 3-D meshes */

/* Put U_el into local coordinates */

			   check = rotXmat2(U_el_local, rotate, U_el, 1, neqel);
			   if(!check) printf( "Problems with rotXmat2 \n");
			}

			check=matX(strain_el, B, U_el_local, sdim, 1, neqel6 );
			if(!check) printf( "Problems with matX \n");

#if 0
			for( i1 = 0; i1 < sdim; ++i1 )
			{
				 printf("%12.8f ",*(strain_el+i1));
				 /*printf("%12.2f ",*(strain_el+i1));
				 printf("%12.8f ",*(B+i1));*/
			}
			printf("\n");
#endif

#if !DEBUG
			*(strain_el) /= 2.0*(*(Arean + k));
			*(strain_el + 1) /= 2.0*(*(Arean + k));
			*(strain_el + 2) /= 2.0*(*(Arean + k));
#endif

/* Update of the global strain matrix */

			strain[k].xx = *(strain_el);
			strain[k].yy = *(strain_el + 1);
			strain[k].xy = *(strain_el + 2);

/* Calculate the principal straines */

			xxaddyy = .5*(strain[k].xx + strain[k].yy);
			xxsubyy = .5*(strain[k].xx - strain[k].yy);
			xysq = strain[k].xy*strain[k].xy;

			strain[k].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			strain[k].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);
			/*printf("%14.6e %14.6e %14.6e\n",xxaddyy,xxsubyy,xysq);*/

/* Calculation of the local stress matrix */

			*(stress_el) = strain[k].xx*D11+
				strain[k].yy*D12;
			*(stress_el+1) = strain[k].xx*D21+
				strain[k].yy*D22;
			*(stress_el+2) = strain[k].xy*Gt;

/* Update of the global stress matrix */

			stress[k].xx += *(stress_el);
			stress[k].yy += *(stress_el+1);
			stress[k].xy += *(stress_el+2);

/* Calculate the principal stresses */

			xxaddyy = .5*(stress[k].xx + stress[k].yy);
			xxsubyy = .5*(stress[k].xx - stress[k].yy);
			xysq = stress[k].xy*stress[k].xy;

			stress[k].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			stress[k].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);

			for( j = 0; j < npel; ++j )
			{
			    node = *(connect+npel*k+j);

/* Add all the strains for a particular node from all the elements which share that node */

			    strain_node[node].xx += strain[k].xx;
			    strain_node[node].yy += strain[k].yy;
			    strain_node[node].xy += strain[k].xy;
			    strain_node[node].I += strain[k].I;
			    strain_node[node].II += strain[k].II;

/* Add all the stresses for a particular node from all the elements which share that node */

			    stress_node[node].xx += stress[k].xx;
			    stress_node[node].yy += stress[k].yy;
			    stress_node[node].xy += stress[k].xy;
			    stress_node[node].I += stress[k].I;
			    stress_node[node].II += stress[k].II;

			}

/*
			printf("%14.6e ",stress[k].xx);
			printf("%14.6e ",stress[k].yy);
			printf("%14.6e ",stress[k].xy);
			printf("%14.6e ",stress[k].I);
			printf("%14.6e ",stress[k].II);
			printf( "\n");
*/

			/*printf( "\n");*/
		}
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
		   strain_node[i].xy /= *(node_counter + i);
		   strain_node[i].I /= *(node_counter + i);
		   strain_node[i].II /= *(node_counter + i);

		   stress_node[i].xx /= *(node_counter + i);
		   stress_node[i].yy /= *(node_counter + i);
		   stress_node[i].xy /= *(node_counter + i);
		   stress_node[i].I /= *(node_counter + i);
		   stress_node[i].II /= *(node_counter + i);
	  }
	}

	return 1;
}


/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on a quad.

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
#include "qdconst.h"
#include "qdstruct.h"

extern int analysis_flag, dof, neqn, numel, numnp, plane_stress_flag, sof, flag_3D,
	local_vec_flag;
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

int matXrot3(double *, double *, double *, int, int);

int rotXmat3(double *, double *, double *, int, int);

int rotTXmat3(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int quadB( double *,double *);

int qdshg( double *, int, double *, double *, double *);

int qdstress_shg( double *, int, double *, double *, double * );

int qdKassemble(double *A, int *connect, double *coord, int *el_matl, double *force,
	int *id, int *idiag, double *K_diag, int *lm, double *local_xyz, MATL *matl,
	double *node_counter, STRAIN *strain, SDIM *strain_node, STRESS *stress,
	SDIM *stress_node, double *U)
{
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness, fdum1, fdum2, fdum3, fdum4;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq64];
	double rotate[npel*nsd2*nsd];
	double force_el[neqel], U_el[neqel], U_el_local[npel*(ndof-1)];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double stress_el[sdim], strain_el[sdim], xxaddyy, xxsubyy, xysq;
	double det[num_int], wXdet;
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd], vec_dum4[nsd];
	double xp[npel], xp0[npel], xp1[npel], xp2[npel],
		yp[npel], yp0[npel], yp1[npel], yp2[npel],
		zp[npel], zp0[npel], zp1[npel], zp2[npel];

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

/* Create the coord_el transpose vector for one element */

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

		memset(rotate,0,npel*nsd2*nsd*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel + j) = *(coord_el_trans + npel*1 + j);
		    }
		}
		else
		{
/* For 3-D quad meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  The local basis used for the
   rotation has already been calculated and stored in local_xyz[].  Below, it is 
   copied to rotate[].

   As can be seen below, there are 2 ways to do the rotation.  The first, implemented when:

       local_vec_flag = 0

   has a basis for all four nodes of every element.  The second, implemented when: 

       local_vec_flag = 1 

   is based on ANSYS where there is only one basis for the element.  You can read more
   about it in:

       ~/slffea-1.4/common/local_vec.c
*/

		    if(!local_vec_flag)
		    {
/* This is the first method. */

			for( j = 0; j < npel; ++j )
			{
			    *(rotate + j*nsd2*nsd) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j);
			    *(rotate + j*nsd2*nsd + 1) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 1);
			    *(rotate + j*nsd2*nsd + 2) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 2);
			    *(rotate + j*nsd2*nsd + 3) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 3);
			    *(rotate + j*nsd2*nsd + 4) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 4);
			    *(rotate + j*nsd2*nsd + 5) =
				*(local_xyz + npel*nsdsq*k + nsdsq*j + 5);

/* Put coord_el into local coordinates */

			    check = matX( (coord_el_local+nsd2*j), (rotate + j*nsd2*nsd),
				(coord_el+nsd*j), nsd2, 1, nsd);
			    if(!check) printf( "Problems with  matX \n");
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}

/* The code below does the same as the matrix multiplication above to calculate the local
   coordinates and is clearer but isn't as efficient.  If the code below is used, make sure
   to comment out both the matrix multiplication above as well as the setting of
   coord_el_trans.

			dum = nsd*npel;
			check = rotXmat3(coord_el_local, rotate, coord_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat3 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}
*/
		    }
		    else
		    {
/* This is the second method. */

			*(rotate)     = *(local_xyz + nsdsq*k);
			*(rotate + 1) = *(local_xyz + nsdsq*k + 1);
			*(rotate + 2) = *(local_xyz + nsdsq*k + 2);
			*(rotate + 3) = *(local_xyz + nsdsq*k + 3);
			*(rotate + 4) = *(local_xyz + nsdsq*k + 4);
			*(rotate + 5) = *(local_xyz + nsdsq*k + 5);

			dum = nsd*npel;
			check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat2 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			}
		    }
		}


/* Assembly of the shg matrix for each integration point */

		check=qdshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with qdshg \n");

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(force_el,0,neqel*sof);
		memset(K_temp,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_local,0,neqlsq64*sof);

/* Assembly of the B matrix */

		    check = quadB((shg+npel*(nsd2+1)*j),B);
		    if(!check) printf( "Problems with quadB \n");

		    for( i1 = 0; i1 < neqel8; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel8*1+i1)*D12;
			*(DB+neqel8*1+i1) = *(B+i1)*D21+
				*(B+neqel8*1+i1)*D22;
			*(DB+neqel8*2+i1) = *(B+neqel8*2+i1)*Gt;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, B, DB, neqel8, neqel8, sdim);
		    if(!check) printf( "Problems with matXT  \n");
		    for( i2 = 0; i2 < neqlsq64; ++i2 )
		    {
			  *(K_el+i2) += *(K_local+i2)*wXdet;
		    }
		}

		if(!flag_3D)
		{
/* For 2-D meshes */
		   for( i = 0; i < npel; ++i )
		   {
			for( j = 0; j < npel; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof*neqel*i + ndof*j) =
				*(K_el + ndof2*neqel8*i + ndof2*j);
			   *(K_temp + ndof*neqel*i + ndof*j + 1) =
				*(K_el + ndof2*neqel8*i + ndof2*j + 1);
			   *(K_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof*neqel*i + neqel + ndof*j) =
				*(K_el + ndof2*neqel8*i + neqel8 + ndof2*j);
			   *(K_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				*(K_el + ndof2*neqel8*i + neqel8 + ndof2*j + 1);
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

		   if(!local_vec_flag)
		   {
			check = matXrot3(K_temp, K_el, rotate, neqel8, neqel);
			if(!check) printf( "Problems with matXrot3 \n");

			check = rotTXmat3(K_el, rotate, K_temp, neqel, neqel);
			if(!check) printf( "Problems with rotTXmat3 \n");
		   }
		   else
		   {
			check = matXrot2(K_temp, K_el, rotate, neqel8, neqel);
			if(!check) printf( "Problems with matXrot2 \n");

			check = rotTXmat2(K_el, rotate, K_temp, neqel, neqel);
			if(!check) printf( "Problems with rotTXmat2 \n");
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

/* Assembly of the qdstress_shg matrix for each nodal point */

/*
   qdstress_shg calculates shg at the nodes.  It is more efficient than qdshg
   because it removes all the zero multiplications from the calculation of shg. 
   You can use either function when calculating shg at the nodes. 

			    check=qdstress_shg(det, k, shl_node2, shg_node, coord_el_local_trans);
			    check=qdshg(det, k, shl_node, shg_node, coord_el_local_trans);
*/

			if(gauss_stress_flag)
			{
/* Calculate shg at integration point */
			    check=qdshg(det, k, shl, shg, coord_el_local_trans);
			    if(!check) printf( "Problems with qdshg \n");
			}
			else
			{
/* Calculate shg at nodal point */
			    check=qdstress_shg(det, k, shl_node2, shg_node, coord_el_local_trans);
			    if(!check) printf( "Problems with qdstress_shg \n");
			}

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

				*(U_el_local + 6) = *(U_el + 9);
				*(U_el_local + 7) = *(U_el + 10);
			}
			else
			{
/* For 3-D meshes */

/* Put U_el into local coordinates */

			    if(!local_vec_flag)
			    {
				check = rotXmat3(U_el_local, rotate, U_el, 1, neqel);
				if(!check) printf( "Problems with rotXmat3 \n");
			    }
			    else
			    {
				check = rotXmat2(U_el_local, rotate, U_el, 1, neqel);
				if(!check) printf( "Problems with rotXmat2 \n");
			    }
			}

			for( j = 0; j < num_int; ++j )
			{

			   memset(B,0,soB*sof);
			   memset(stress_el,0,sdim*sof);
			   memset(strain_el,0,sdim*sof);

			   node = *(connect+npel*k+j);

/* Assembly of the B matrix */

			   if(gauss_stress_flag)
			   {
/* Calculate B matrix at integration point */
				check = quadB((shg+npel*(nsd2+1)*j),B);
				if(!check) printf( "Problems with quadB \n");
			   }
			   else
			   {
/* Calculate B matrix at nodal point */
				check = quadB((shg_node+npel*(nsd2+1)*j),B);
				if(!check) printf( "Problems with quadB \n");
			   }
		
/* Calculation of the local strain matrix */

			   check=matX(strain_el, B, U_el_local, sdim, 1, neqel8 );
			   if(!check) printf( "Problems with matX \n");

#if 0
			   for( i1 = 0; i1 < sdim; ++i1 )
			   {
				printf("%12.8f ",*(stress_el+i1));
				/*printf("%12.2f ",*(stress_el+i1));
				printf("%12.8f ",*(B+i1));*/
			   }
			   printf("\n");
#endif

/* Update of the global strain matrix */

			   strain[k].pt[j].xx = *(strain_el);
			   strain[k].pt[j].yy = *(strain_el+1);
			   strain[k].pt[j].xy = *(strain_el+2);

/* Calculate the principal straines */

			   xxaddyy = .5*(strain[k].pt[j].xx + strain[k].pt[j].yy);
			   xxsubyy = .5*(strain[k].pt[j].xx - strain[k].pt[j].yy);
			   xysq = strain[k].pt[j].xy*strain[k].pt[j].xy;

			   strain[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			   strain[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);
			   /*printf("%14.6e %14.6e %14.6e\n",xxaddyy,xxsubyy,xysq);*/

/* Add all the strains for a particular node from all the elements which share that node */

			   strain_node[node].xx += strain[k].pt[j].xx;
			   strain_node[node].yy += strain[k].pt[j].yy;
			   strain_node[node].xy += strain[k].pt[j].xy;
			   strain_node[node].I += strain[k].pt[j].I;
			   strain_node[node].II += strain[k].pt[j].II;

/* Calculation of the local stress matrix */

			   *(stress_el) = strain[k].pt[j].xx*D11+
				strain[k].pt[j].yy*D12;
			   *(stress_el+1) = strain[k].pt[j].xx*D21+
				strain[k].pt[j].yy*D22;
			   *(stress_el+2) = strain[k].pt[j].xy*Gt;

/* Update of the global stress matrix */

			   stress[k].pt[j].xx += *(stress_el);
			   stress[k].pt[j].yy += *(stress_el+1);
			   stress[k].pt[j].xy += *(stress_el+2);

/* Calculate the principal stresses */

			   xxaddyy = .5*(stress[k].pt[j].xx + stress[k].pt[j].yy);
			   xxsubyy = .5*(stress[k].pt[j].xx - stress[k].pt[j].yy);
			   xysq = stress[k].pt[j].xy*stress[k].pt[j].xy;

			   stress[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			   stress[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);

/* Add all the stresses for a particular node from all the elements which share that node */

			   stress_node[node].xx += stress[k].pt[j].xx;
			   stress_node[node].yy += stress[k].pt[j].yy;
			   stress_node[node].xy += stress[k].pt[j].xy;
			   stress_node[node].I += stress[k].pt[j].I;
			   stress_node[node].II += stress[k].pt[j].II;

/*
			   printf("%14.6e ",stress[k].pt[j].xx);
			   printf("%14.6e ",stress[k].pt[j].yy);
			   printf("%14.6e ",stress[k].pt[j].xy);
			   printf( "\n");
*/
			}
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


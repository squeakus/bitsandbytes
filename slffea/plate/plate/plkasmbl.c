/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on plate elements made
    up of 4 node quadrilaterals.

		Updated 10/23/06

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
#include "plconst.h"
#include "plstruct.h"

extern int analysis_flag, dof, neqn, numel, numnp, sof, plane_stress_flag, flag_3D;
extern int gauss_stress_flag;
extern int LU_decomp_flag, numel_K, numel_P;
extern SH shg, shg_node, shl, shl_node;
extern double shl_node2[sosh_node2], w[num_int+1], *Area0;

int cubic( double *);

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int rotTXmat(double *, double *, double *, int, int);

int matXrot4(double *, double *, double *, int, int);

int rotXmat4(double *, double *, double *, int, int);

int rotTXmat4(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int plateB4pt( double *, double *);

int plateB4pt_node( double *, double *);

int plateB1pt( double *, double *);

int quadB( double *,double *);

int plshg( double *, int, SH, SH, double *);

int plstress_shg( double *, int, double *, double *, double * );

int plKassemble(double *A, int *connect, double *coord, CURVATURE *curve, MDIM *curve_node,
	int *el_matl, double *force, int *id, int *idiag, double *K_diag, int *lm,
	double *local_xyz, MATL *matl, MOMENT *moment, MDIM *moment_node, double *node_counter,
	STRAIN *strain, SDIM *strain_node, STRESS *stress, SDIM *stress_node, double *U,
	double *Uz_fib)
{
	int i, i1, i2, i3, j, k, dof_el[neqel20], dof_el6[neqel24], sdof_el[npel4*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G1, G2, G3, thickness, shearK, const1, const2;
	double fdum1, fdum2, fdum3, fdum4;
	double lamda, mu;
	double D11,D12,D21,D22,D11mem,D12mem,D21mem,D22mem,Gmem;
	double B[soB], DB[soB], Bmem[soBmem], DBmem[soBmem];
	double K_temp[neqlsq576], K_el[neqlsq576], K_local[neqlsq144];
	double K_bend[neqlsq144], K_mem[neqlsq64];
	double rotate[nsdsq];
	double force_el[neqel24], force_temp[neqel24];
	double U_el[neqel24], U_el_local[neqel24], U_el_mem_local[neqel8];
	double coord_el[npel4*nsd], coord_el_trans[npel4*nsd],
		coord_el_local[npel4*nsd2], coord_el_local_trans[npel4*nsd2];
	double stress_el[sdim5], strain_el[sdim5], xxaddyy, xxsubyy, xysq, invariant[nsd],
		yzsq, zxsq, xxyy;
	double det[num_int+1], wXdet;

	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;
		shearK = matl[matl_num].shear;

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* The lamda below is for plane stress */

		if(plane_stress_flag) lamda = Emod*Pois/(1.0-Pois*Pois);

		/*printf("lamda, mu, Emod, Pois %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* FOR THE MEMBRANE */

		D11mem = thickness*(lamda+2.0*mu);
		D12mem = thickness*lamda;
		D21mem = thickness*lamda;
		D22mem = thickness*(lamda+2.0*mu);

		Gmem = thickness*mu;

/* FOR THE BENDING */

		const1 = thickness*thickness*thickness/12.0;
		const2 = thickness*shearK;

		D11 = (lamda+2.0*mu)*const1;
		D12 = lamda*const1;
		D21 = lamda*const1;
		D22 = (lamda+2.0*mu)*const1;
		G1 = mu*const1;
		G2 = mu*const2;
		G3 = mu*const2;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel4*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel4*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el+ndof5*j) = ndof5*node;
			*(dof_el+ndof5*j+1) = ndof5*node+1;
			*(dof_el+ndof5*j+2) = ndof5*node+2;
			*(dof_el+ndof5*j+3) = ndof5*node+3;
			*(dof_el+ndof5*j+4) = ndof5*node+4;

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;

/* Count the number of times a particular node is part of an element */

			if(analysis_flag == 1)
				*(node_counter + node) += 1.0;
		}

		memset(rotate,0,nsdsq*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel4 + j) = *(coord_el_trans + npel4*1 + j);
		    }
		}
		else
		{
/* For 3-D quad meshes, I have to rotate from the global coordinates to the local x and
   y coordinates which lie in the plane of the element.  The local basis used for the
   rotation has already been calculated and stored in local_xyz[].  Below, it is
   copied to rotate[].

   The rotation matrix, rotate[], is used to rotate the global U_el, made up of
   nodal displacements x, y, and z and rotation angels x, y, and z into the local x, y,
   and z coordinates of the plane of the plate.  It should be noted that the rotation
   angle z doesn't really exist, but is added to make the rotation multiplications easier.
   I can now take advantage of the beam and truss rotation functions by doing this.
   This rotation is also used for the rotations of the element stiffnesses.  
*/
		    *(rotate)     = *(local_xyz + nsdsq*k);
		    *(rotate + 1) = *(local_xyz + nsdsq*k + 1);
		    *(rotate + 2) = *(local_xyz + nsdsq*k + 2);
		    *(rotate + 3) = *(local_xyz + nsdsq*k + 3);
		    *(rotate + 4) = *(local_xyz + nsdsq*k + 4);
		    *(rotate + 5) = *(local_xyz + nsdsq*k + 5);
		    *(rotate + 6) = *(local_xyz + nsdsq*k + 6);
		    *(rotate + 7) = *(local_xyz + nsdsq*k + 7);
		    *(rotate + 8) = *(local_xyz + nsdsq*k + 8);

/* Rotate the global coordinates into the local x and y coordinates of the plane of the plate.
*/
		    dum = nsd*npel4;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel4*1 + j) = *(coord_el_local+nsd2*j+1);
		    }
		}

/* Assembly of the shg matrix for each integration point */

		check=plshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with plshg \n");

		memset(U_el,0,neqel24*sof);
		memset(K_el,0,neqlsq576*sof);
		memset(force_el,0,neqel24*sof);
		memset(K_temp,0,neqlsq576*sof);
		memset(K_bend,0,neqlsq144*sof);
		memset(K_mem,0,neqlsq64*sof);

/* Calculate the lower part of the B and DB matrices for 1X1 point gaussian integration */

		memset((B+neqel12*(sdim5-2)),0,neqel12*(sdim5-3)*sof);
		memset((DB+neqel12*(sdim5-2)),0,neqel12*(sdim5-3)*sof);

		check = plateB1pt(shg.shear, (B+(sdim5-2)*neqel12));
		if(!check) printf( "Problems with plateB1pt \n");
		for( i1 = 0; i1 < neqel12; ++i1 )
		{
			*(DB+neqel12*3+i1) = *(B+neqel12*3+i1)*G2;
			*(DB+neqel12*4+i1) = *(B+neqel12*4+i1)*G3;
		}

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		for( j = 0; j < num_int; ++j )
		{

/* FOR THE MEMBRANE */

		    memset(Bmem,0,soBmem*sof);
		    memset(DBmem,0,soBmem*sof);
		    memset(K_local,0,neqlsq64*sof);

/* Assembly of the Bmem matrix */

		    check = quadB((shg.bend+npel4*(nsd2+1)*j), Bmem);
		    if(!check) printf( "Problems with quadB \n");

		    for( i1 = 0; i1 < neqel8; ++i1 )
		    {
			*(DBmem+i1) = *(Bmem+i1)*D11mem+
				*(Bmem+neqel8*1+i1)*D12mem;
			*(DBmem+neqel8*1+i1) = *(Bmem+i1)*D21mem+
				*(Bmem+neqel8*1+i1)*D22mem;
			*(DBmem+neqel8*2+i1) = *(Bmem+neqel8*2+i1)*Gmem;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, Bmem, DBmem, neqel8, neqel8, sdim3);
		    if(!check) printf( "Problems with matXT  \n");
		    for( i2 = 0; i2 < neqlsq64; ++i2 )
		    {
			  *(K_mem+i2) += *(K_local+i2)*wXdet;
		    }


/* FOR THE BENDING */

		    memset(B,0,neqel12*(sdim5-2)*sof);
		    memset(DB,0,neqel12*(sdim5-2)*sof);
		    memset(K_local,0,neqlsq144*sof);

/* Calculate the upper part of the B and DB matrices for 4X4 point gaussian integration */

		    check = plateB4pt((shg.bend+npel4*(nsd2+1)*j), B);
		    if(!check) printf( "Problems with plateB4pt \n");

		    for( i1 = 0; i1 < neqel12; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11 +
				*(B+neqel12*1+i1)*D12;
			*(DB+neqel12*1+i1) = *(B+i1)*D21 +
				*(B+neqel12*1+i1)*D22;
			*(DB+neqel12*2+i1) = *(B+neqel12*2+i1)*G1;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(K_local, B, DB, neqel12, neqel12, sdim5);
		    if(!check) printf( "Problems with matXT \n");

/* Note that I'm using the 4X4 determinant and weight for the 1X1 Gauss integration
   which is added 4 times.  This is a valid operation which can be proven. */

		    for( i2 = 0; i2 < neqlsq144; ++i2 )
		    {
			  *(K_bend+i2) += *(K_local+i2)*wXdet;
		    }
		}

/*
		for( i1 = 0; i1 < neqel8; ++i1 )
		{
		   for( i2 = 0; i2 < neqel8; ++i2 )
		   {
			printf("%12.5e ",*(K_mem+neqel8*i1+i2));
		   }
		   printf("\n ");
		}
		printf("\n ");
*/

/*
		for( i1 = 0; i1 < neqel12; ++i1 )
		{
		   for( i2 = 0; i2 < neqel12; ++i2 )
		   {
			printf("%12.5e ",*(K_bend+neqel12*i1+i2));
		   }
		   printf("\n ");
		}
		printf("\n ");
*/

/* Unlike the quadrilateral and triangle membrane elements, the lines below are executed
   for both 2-D and 3-D elements because I wanted to make the rotation easier to understand.
   This element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out K_bend to test if I can get the same results
   as the quadrilateral element which has no bending.   Also, I need to zero out M_bend in
   "plmasmbl.c" with:

                memset(M_bend,0,neqlsq144*sof);

   Note that if an eigenmode comparison is made, I must change the num_eigen in "fempl.c" to:

      num_eigen = (int)(2.0*nmode);

   to match the quad.
*/
#if 0
		memset(K_bend,0,neqlsq144*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel4; ++i )
		{
			for( j = 0; j < npel4; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof6*neqel24*i + ndof6*j) =
				*(K_mem + ndof2*neqel8*i + ndof2*j);
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 1) =
				*(K_mem + ndof2*neqel8*i + ndof2*j + 1);
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j) =
				*(K_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j);
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 1) =
				*(K_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j + 1);
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 5) = 0.0;
			}
/* The line below is an attempt to add an artificial stiffness to the rotation in local z, but
   I think my results are better without it.
*/
			/* *(K_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*i + 5) = 10.0e20;*/
		}
		memcpy(K_el, K_temp, neqlsq576*sizeof(double));

		/*for( i1 = 0; i1 < neqel24; ++i1 )
		{
		   for( i2 = 0; i2 < neqel24; ++i2 )
		   {
			printf("%12.5e ",*(K_temp+neqel24*i1+i2));
		   }
		   printf("\n");
		}
		printf("aaaa %d %d\n", neqel24, k);*/


		if(flag_3D)
		{
/* For 3-D meshes */

/* Put K back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

#if 0
		   check = matXrot4(K_temp, K_el, rotate, neqel24, neqel24);
		   if(!check) printf( "Problems with matXrot4 \n");
#endif
		   check = matXrot(K_temp, K_el, rotate, neqel24, neqel24);
		   if(!check) printf( "Problems with matXrot \n");

		/*for( i1 = 0; i1 < neqel24; ++i1 )
		{
		   for( i2 = 0; i2 < neqel24; ++i2 )
		   {
			printf("%12.5e ",*(K_temp+neqel24*i1+i2));
		   }
		   printf("\n");
		}
		printf("bbbb %d %d\n", neqel24, k);*/

#if 0
		   check = rotTXmat4(K_el, rotate, K_temp, neqel24, neqel24);
		   if(!check) printf( "Problems with rotTXmat4 \n");
#endif
		   check = rotTXmat(K_el, rotate, K_temp, neqel24, neqel24);
		   if(!check) printf( "Problems with rotTXmat \n");
		}

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6 + j));
		}

		check = matX(force_el, K_el, U_el, neqel24, 1, neqel24);
		if(!check) printf( "Problems with matX \n");

		/*for( i1 = 0; i1 < neqel24; ++i1 )
		{
		   for( i2 = 0; i2 < neqel24; ++i2 )
		   {
			printf("%12.5e ",*(K_el+neqel24*i1+i2));
		   }
		   printf("\n");
		}
		printf("cccc %d %d\n", neqel24, k);*/

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel24; ++j )
			{
				*(force + *(dof_el6+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(A, idiag, K_el, (lm + k*neqel24),
				neqel24);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(A, dof_el6, k, K_diag, K_el,
				neqel24, neqlsq576, numel_K);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel24; ++j )
			{
				*(force + *(dof_el6+j)) += *(force_el + j);
			}

/* Calculate the element stresses */

			const2 = shearK;
			G2 = mu*const2;
			G3 = mu*const2;

/*
   plstress_shg calculates shg at the nodes.  It is more efficient than plshg
   because it removes all the zero multiplications from the calculation of shg.
   You can use either function when calculating shg at the nodes.

			   check=plstress_shg(det, k, shl_node2, shg_node.bend, coord_el_local_trans);
			   check=plshg(det, k, shl_node, shg_node, coord_el_local_trans);
*/

			if(gauss_stress_flag)
			{
/* Calculate shg at integration point */
			   check=plshg(det, k, shl, shg, coord_el_local_trans);
			   if(!check) printf( "Problems with plshg \n");
			}
			else
			{
/* Calculate shg at nodal point */
			   check=plstress_shg(det, k, shl_node2, shg_node.bend, coord_el_local_trans);
			   if(!check) printf( "Problems with plshg \n");
			}

/* Calculation of the local strain matrix */

/* Determine local U coordinates */

			if(!flag_3D)
			{
/* For 2-D meshes */
				*(U_el_local) = *(U_el + 2);
				*(U_el_local + 1) = *(U_el + 3);
				*(U_el_local + 2) = *(U_el + 4);

				*(U_el_local + 3) = *(U_el + 8);
				*(U_el_local + 4) = *(U_el + 9);
				*(U_el_local + 5) = *(U_el + 10);

				*(U_el_local + 6) = *(U_el + 14);
				*(U_el_local + 7) = *(U_el + 15);
				*(U_el_local + 8) = *(U_el + 16);

				*(U_el_local + 9) = *(U_el + 20);
				*(U_el_local + 10) = *(U_el + 21);
				*(U_el_local + 11) = *(U_el + 22);
			}
			else
			{
/* For 3-D meshes */

/* Put U_el into local coordinates */

#if 0
			   check = rotXmat4(U_el_local, rotate, U_el, 1, neqel24);
			   if(!check) printf( "Problems with rotXmat4 \n");
#endif
			   check = rotXmat(U_el_local, rotate, U_el, 1, neqel24);
			   if(!check) printf( "Problems with rotXmat4 \n");

/* FOR THE MEMBRANE */

			   *(U_el_mem_local) = *(U_el_local);
			   *(U_el_mem_local + 1) = *(U_el_local + 1);

			   *(U_el_mem_local + 2) = *(U_el_local + 6);
			   *(U_el_mem_local + 3) = *(U_el_local + 7);

			   *(U_el_mem_local + 4) = *(U_el_local + 12);
			   *(U_el_mem_local + 5) = *(U_el_local + 13);

			   *(U_el_mem_local + 6) = *(U_el_local + 18);
			   *(U_el_mem_local + 7) = *(U_el_local + 19);

/* FOR THE BENDING */

			   *(U_el) = *(U_el_local + 2);
			   *(U_el + 1) = *(U_el_local + 3);
			   *(U_el + 2) = *(U_el_local + 4);

			   *(U_el + 3) = *(U_el_local + 8);
			   *(U_el + 4) = *(U_el_local + 9);
			   *(U_el + 5) = *(U_el_local + 10);

			   *(U_el + 6) = *(U_el_local + 14);
			   *(U_el + 7) = *(U_el_local + 15);
			   *(U_el + 8) = *(U_el_local + 16);

			   *(U_el + 9) = *(U_el_local + 20);
			   *(U_el + 10) = *(U_el_local + 21);
			   *(U_el + 11) = *(U_el_local + 22);

/* Set the local z displacement vector to the Uz lamina vector.  My intention
   was to use Uz_fib to adjust the displacements in the global x and y directions
   like is done for the shell.  When I printed out Uz_fib for the different elements
   though, I saw that the values of Uz_fib were different for the node depending on
   the element.  This is expected since Uz_fib represents the displacement in the
   normal direction of the plate face at a node, and each element has its own normal
   face direction.
*/

			   *(Uz_fib + *(connect+npel4*k)) = *(U_el_local + 2);
			   *(Uz_fib + *(connect+npel4*k + 1)) = *(U_el_local + 8);
			   *(Uz_fib + *(connect+npel4*k + 2)) = *(U_el_local + 14);
			   *(Uz_fib + *(connect+npel4*k + 3)) = *(U_el_local + 20);

			   memcpy(U_el_local, U_el, neqel12*sizeof(double));
			}


			memset((B+neqel12*(sdim5-2)),0,neqel12*(sdim5-3)*sof);

/* Assembly of the B matrix */

			check = plateB1pt(shg.shear, (B+(sdim5-2)*neqel12));
			if(!check) printf( "Problems with plateB1pt \n");

			for( j = 0; j < num_int; ++j )
			{
			   node = *(connect+npel4*k+j);

/* FOR THE MEMBRANE */

			   memset(Bmem,0,soBmem*sof);
			   memset(stress_el,0,sdim3*sof);
			   memset(strain_el,0,sdim3*sof);

			   node = *(connect+npel4*k+j);

/* Assembly of the Bmem matrix */

			   if(gauss_stress_flag)
			   {
/* Calculate Bmem matrix at integration point */
				check = quadB((shg.bend+npel4*(nsd2+1)*j),Bmem);
				if(!check) printf( "Problems with quadB \n");
			   }
			   else
			   {
/* Calculate Bmem matrix at nodal point */
				check = quadB((shg_node.bend+npel4*(nsd2+1)*j),Bmem);
				if(!check) printf( "Problems with quadB \n");
			   }

/* Calculation of the local strain matrix */

			   check=matX(strain_el, Bmem, U_el_mem_local, sdim3, 1, neqel8 );
			   if(!check) printf( "Problems with matX \n");

/* Update of the global strain matrix */

			   strain[k].pt[j].xx = *(strain_el);
			   strain[k].pt[j].yy = *(strain_el+1);
			   strain[k].pt[j].xy = *(strain_el+2);

/* Add all the strains for a particular node from all the elements which share that node */

			   strain_node[node].xx += strain[k].pt[j].xx;
			   strain_node[node].yy += strain[k].pt[j].yy;
			   strain_node[node].xy += strain[k].pt[j].xy;

/* Calculation of the local stress matrix */

			   *(stress_el) = strain[k].pt[j].xx*D11mem +
				strain[k].pt[j].yy*D12mem;
			   *(stress_el+1) = strain[k].pt[j].xx*D21mem +
				strain[k].pt[j].yy*D22mem;
			   *(stress_el+2) = strain[k].pt[j].xy*Gmem;

/* Update of the global stress matrix */

			   stress[k].pt[j].xx += *(stress_el);
			   stress[k].pt[j].yy += *(stress_el+1);
			   stress[k].pt[j].xy += *(stress_el+2);

/* Add all the stresses for a particular node from all the elements which share that node */

			   stress_node[node].xx += stress[k].pt[j].xx;
			   stress_node[node].yy += stress[k].pt[j].yy;
			   stress_node[node].xy += stress[k].pt[j].xy;




/* FOR THE BENDING */
			   memset(B,0,neqel12*(sdim5-2)*sof);
			   memset(stress_el,0,sdim5*sof);
			   memset(strain_el,0,sdim5*sof);


/* Assembly of the B matrix */

			   if(gauss_stress_flag)
			   {
				check = plateB4pt((shg.bend+npel4*(nsd2+1)*j), B);
				if(!check) printf( "Problems with plateB4pt \n");
			   }
			   else
			   {
/* Calculate the shear terms in B at nodes using shg_node */
				check = plateB4pt_node((shg_node.bend+npel4*(nsd2+1)*j), B);	
				if(!check) printf( "Problems with plateB4pt_node \n");
			   }

/* Calculation of the local strain matrix */

			   check=matX(strain_el, B, U_el_local, sdim5, 1, neqel12 );
			   if(!check) printf( "Problems with matX \n");

#if 0
			   for( i1 = 0; i1 < sdim5; ++i1 )
			   {
				   printf("%12.8f ",*(stress_el+i1));
				   /*printf("%12.2f ",*(stress_el+i1));
				   printf("%12.8f ",*(B+i1));*/
			   }
			   printf("\n");
#endif

/* Update of the global strain matrix */

			   curve[k].pt[j].xx = *(strain_el);
			   curve[k].pt[j].yy = *(strain_el+1);
			   curve[k].pt[j].xy = *(strain_el+2);
			   strain[k].pt[j].zx = *(strain_el+3);
			   strain[k].pt[j].yz = *(strain_el+4);

/* Calculate the principal straines */

			   xxaddyy = .5*(curve[k].pt[j].xx + curve[k].pt[j].yy);
			   xxsubyy = .5*(curve[k].pt[j].xx - curve[k].pt[j].yy);
			   xysq = curve[k].pt[j].xy*curve[k].pt[j].xy;

			   curve[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			   curve[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);
			   /*printf("%14.6e %14.6e %14.6e\n",xxaddyy,xxsubyy,xysq);*/

/* Add all the curvatures and strains for a particular node from all the elements which
   share that node */

			   curve_node[node].xx += curve[k].pt[j].xx;
			   curve_node[node].yy += curve[k].pt[j].yy;
			   curve_node[node].xy += curve[k].pt[j].xy;
			   strain_node[node].zx += strain[k].pt[j].zx;
			   strain_node[node].yz += strain[k].pt[j].yz;
			   curve_node[node].I += curve[k].pt[j].I;
			   curve_node[node].II += curve[k].pt[j].II;

/* Calculation of the local stress matrix */

			   *(stress_el)=curve[k].pt[j].xx*D11 +
				curve[k].pt[j].yy*D12;
			   *(stress_el+1)=curve[k].pt[j].xx*D21 +
				curve[k].pt[j].yy*D22;
			   *(stress_el+2)=curve[k].pt[j].xy*G1;
			   *(stress_el+3)=strain[k].pt[j].zx*G2;
			   *(stress_el+4)=strain[k].pt[j].yz*G3;

/* Update of the global stress matrix */

			   moment[k].pt[j].xx += *(stress_el);
			   moment[k].pt[j].yy += *(stress_el+1);
			   moment[k].pt[j].xy += *(stress_el+2);
			   stress[k].pt[j].zx += *(stress_el+3);
			   stress[k].pt[j].yz += *(stress_el+4);

/* Calculate the principal stresses */

			   xxaddyy = .5*(moment[k].pt[j].xx + moment[k].pt[j].yy);
			   xxsubyy = .5*(moment[k].pt[j].xx - moment[k].pt[j].yy);
			   xysq = moment[k].pt[j].xy*moment[k].pt[j].xy;

			   moment[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			   moment[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);

/* Add all the moments and stresses for a particular node from all the elements which
   share that node */

			   moment_node[node].xx += moment[k].pt[j].xx;
			   moment_node[node].yy += moment[k].pt[j].yy;
			   moment_node[node].xy += moment[k].pt[j].xy;
			   stress_node[node].zx += stress[k].pt[j].zx;
			   stress_node[node].yz += stress[k].pt[j].yz;
			   moment_node[node].I += moment[k].pt[j].I;
			   moment_node[node].II += moment[k].pt[j].II;

/*
			   printf("%14.6e ", moment_node[node].xx);
			   printf("%14.6e ", moment_node[node].yy);
			   printf("%14.6e ", moment_node[node].xy);
			   printf( "\n");
*/

/* The principal strains involve both bending (stress zx and yz) and membrane
   (stress xx, yy, xy )components.
*/

/* Calculate the principal straines */

			   memset(invariant,0,nsd*sof);
			   xysq = strain[k].pt[j].xy*strain[k].pt[j].xy;
			   zxsq = strain[k].pt[j].zx*strain[k].pt[j].zx;
			   yzsq = strain[k].pt[j].yz*strain[k].pt[j].yz;
			   xxyy = strain[k].pt[j].xx*strain[k].pt[j].yy;

			   *(invariant) = - strain[k].pt[j].xx -
				strain[k].pt[j].yy;
			   *(invariant+1) = xxyy - yzsq - zxsq - xysq;
			   *(invariant+2) = -
				2*strain[k].pt[j].yz*strain[k].pt[j].zx*strain[k].pt[j].xy +
				yzsq*strain[k].pt[j].xx + zxsq*strain[k].pt[j].yy;
				
			   check = cubic(invariant);

			   strain[k].pt[j].I = *(invariant);
			   strain[k].pt[j].II = *(invariant+1);
			   strain[k].pt[j].III = *(invariant+2);

			   strain_node[node].I += strain[k].pt[j].I;
			   strain_node[node].II += strain[k].pt[j].II;
			   strain_node[node].III += strain[k].pt[j].III;

/* Calculate the principal stresses */

			   memset(invariant,0,nsd*sof);
			   xysq = stress[k].pt[j].xy*stress[k].pt[j].xy;
			   zxsq = stress[k].pt[j].zx*stress[k].pt[j].zx;
			   yzsq = stress[k].pt[j].yz*stress[k].pt[j].yz;
			   xxyy = stress[k].pt[j].xx*stress[k].pt[j].yy;

			   *(invariant) = - stress[k].pt[j].xx -
				stress[k].pt[j].yy;
			   *(invariant+1) = xxyy - yzsq - zxsq - xysq;
			   *(invariant+2) = -
				2*stress[k].pt[j].yz*stress[k].pt[j].zx*stress[k].pt[j].xy +
				yzsq*stress[k].pt[j].xx + zxsq*stress[k].pt[j].yy;

			   check = cubic(invariant);

			   stress[k].pt[j].I = *(invariant);
			   stress[k].pt[j].II = *(invariant+1);
			   stress[k].pt[j].III = *(invariant+2);

/* Add all the stresses for a particular node from all the elements which share that node */

			   stress_node[node].I += stress[k].pt[j].I;
			   stress_node[node].II += stress[k].pt[j].II;
			   stress_node[node].III += stress[k].pt[j].III;

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

/* Average all the moments, stresses, curvatures, and strains at the nodes */

	  for( i = 0; i < numnp ; ++i )
	  {

/* FOR THE MEMBRANE */
		   strain_node[i].xx /= *(node_counter + i);
		   strain_node[i].yy /= *(node_counter + i);
		   strain_node[i].xy /= *(node_counter + i);

		   stress_node[i].xx /= *(node_counter + i);
		   stress_node[i].yy /= *(node_counter + i);
		   stress_node[i].xy /= *(node_counter + i);

/* FOR THE BENDING */
		   curve_node[i].xx /= *(node_counter + i);
		   curve_node[i].yy /= *(node_counter + i);
		   curve_node[i].xy /= *(node_counter + i);
		   strain_node[i].zx /= *(node_counter + i);
		   strain_node[i].yz /= *(node_counter + i);
		   curve_node[i].I /= *(node_counter + i);
		   curve_node[i].II /= *(node_counter + i);

		   moment_node[i].xx /= *(node_counter + i);
		   moment_node[i].yy /= *(node_counter + i);
		   moment_node[i].xy /= *(node_counter + i);
		   stress_node[i].zx /= *(node_counter + i);
		   stress_node[i].yz /= *(node_counter + i);
		   moment_node[i].I /= *(node_counter + i);
		   moment_node[i].II /= *(node_counter + i);

/* FOR THE MEMBRANE AND BENDING */
		   strain_node[i].I /= *(node_counter + i);
		   strain_node[i].II /= *(node_counter + i);
		   strain_node[i].III /= *(node_counter + i);
		   stress_node[i].I /= *(node_counter + i);
		   stress_node[i].II /= *(node_counter + i);
		   stress_node[i].III /= *(node_counter + i);
	  }
	}

	return 1;
}


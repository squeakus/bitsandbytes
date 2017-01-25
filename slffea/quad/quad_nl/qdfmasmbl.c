/*
    This utility function assembles the Mass and force matrix for a finite
    element program which does analysis on a quad which can behave
    nonlinearly.  The mass is used both in the dynamic relaxation
    method as well as the conjugate gradient method.  It is not really
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

    Note that I do not use Bh for DB in qd2Passemble when the conjugate
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
#include "../quad/qdconst.h"
#include "../quad/qdstruct.h"

extern int numel, numnp, dof, sof, plane_stress_flag, flag_3D, local_vec_flag;
extern double shg[sosh], shgh[sosh], shl[sosh], w[num_int], *Area0;

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

int local_vectors( int *, double *, double * );

int qdFMassemble( int *connect, double *coord, double *coordh, int *el_matl,
	double *force, double *mass, double *local_xyz, double *localh_xyz,
	MATL *matl, double *U)
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness, fdum1, fdum2, fdum3, fdum4;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], Bh[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq64];
	double rotate[npel*nsd2*nsd], rotateh[npel*nsd2*nsd];
	double force_el[neqel], U_el[neqel], U_el_local[npel*(ndof-1)];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double coordh_el[npel*nsd], coordh_el_trans[npel*nsd],
		coordh_el_local[npel*nsd2], coordh_el_local_trans[npel*nsd2];
	double stress_el[sdim], strain_el[sdim], xxaddyy, xxsubyy, xysq;
	double det[num_int], wXdet, deth[num_int], wXdeth;
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd],
		vec_dum1[nsd], vec_dum2[nsd], vec_dum3[nsd], vec_dum4[nsd];
	double xp[npel], xp0[npel], xp1[npel], xp2[npel],
		yp[npel], yp0[npel], yp1[npel], yp2[npel],
		zp[npel], zp0[npel], zp1[npel], zp2[npel];
	double mass_el[neqel];

/* initialize all variables  */
	memset(mass,0,dof*sof);

	if(flag_3D)
	{
	    check = local_vectors( connect, coord, local_xyz );
	    if(!check) printf( " Problems with local_vectors \n");

	    check = local_vectors( connect, coordh, localh_xyz );
	    if(!check) printf( " Problems with local_vectors \n");
	}

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

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;

/* Create the coord and coord_trans vector for one element */

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));
			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
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

		memset(rotate,0,npel*nsd2*nsd*sof);
		memset(rotateh,0,npel*nsd2*nsd*sof);

		if(!flag_3D)
		{
		    for( j = 0; j < npel; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel + j) = *(coord_el_trans + npel*1 + j);
			*(coordh_el_local_trans + j) = *(coordh_el_trans + j);
			*(coordh_el_local_trans + 1*npel + j) = *(coordh_el_trans + npel*1 + j);
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

			    *(rotateh + j*nsd2*nsd) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j);
			    *(rotateh + j*nsd2*nsd + 1) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j + 1);
			    *(rotateh + j*nsd2*nsd + 2) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j + 2);
			    *(rotateh + j*nsd2*nsd + 3) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j + 3);
			    *(rotateh + j*nsd2*nsd + 4) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j + 4);
			    *(rotateh + j*nsd2*nsd + 5) =
				*(localh_xyz + npel*nsdsq*k + nsdsq*j + 5);

/* Put coord_el into local coordinates */

			    check = matX( (coord_el_local+nsd2*j), (rotate + j*nsd2*nsd),
				(coord_el+nsd*j), nsd2, 1, nsd);
			    if(!check) printf( "Problems with  matX \n");
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);

/* Put coordh_el into local coordinates */

			    check = matX( (coordh_el_local+nsd2*j), (rotateh + j*nsd2*nsd),
				(coordh_el+nsd*j), nsd2, 1, nsd);
			    if(!check) printf( "Problems with  matX \n");
			    *(coordh_el_local_trans + j) = *(coordh_el_local+nsd2*j);
			    *(coordh_el_local_trans + npel*1 + j) = *(coordh_el_local+nsd2*j+1);
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

			dum = nsd*npel;
			check = rotXmat3(coordh_el_local, rotateh, coordh_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat3 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coordh_el_local_trans + j) = *(coordh_el_local+nsd2*j);
			    *(coordh_el_local_trans + npel*1 + j) = *(coordh_el_local+nsd2*j+1);
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

			*(rotateh)     = *(localh_xyz + nsdsq*k);
			*(rotateh + 1) = *(localh_xyz + nsdsq*k + 1);
			*(rotateh + 2) = *(localh_xyz + nsdsq*k + 2);
			*(rotateh + 3) = *(localh_xyz + nsdsq*k + 3);
			*(rotateh + 4) = *(localh_xyz + nsdsq*k + 4);
			*(rotateh + 5) = *(localh_xyz + nsdsq*k + 5);

			dum = nsd*npel;
			check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat2 \n");
			check = rotXmat2(coordh_el_local, rotateh, coordh_el, 1, dum);
			if(!check) printf( "Problems with  rotXmat2 \n");
			for( j = 0; j < npel; ++j )
			{
			    *(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			    *(coord_el_local_trans + npel*1 + j) = *(coord_el_local+nsd2*j+1);
			    *(coordh_el_local_trans + j) = *(coordh_el_local+nsd2*j);
			    *(coordh_el_local_trans + npel*1 + j) = *(coordh_el_local+nsd2*j+1);
			}
		    }
		}


/* Assembly of the shg matrix for each integration point at 1/2 time */

		check = qdshg(deth, k, shl, shgh, coordh_el_local_trans);
		if(!check) printf( "Problems with brshg \n");

/* Assembly of the shg matrix for each integration point at full time */

		check = qdshg(det, k, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with qdshg \n");

/* The loop over j below calculates the 4 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);
		memset(force_el,0,neqel*sof);
		memset(K_temp,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(Bh,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_local,0,neqlsq64*sof);

/* Assembly of the Bh matrix at 1/2 time */

		    check = quadB((shgh+npel*(nsd2+1)*j),Bh);
		    if(!check) printf( "Problems with quadB \n");

/* Assembly of the B matrix at full time */

		    check = quadB((shg+npel*(nsd2+1)*j),B);
		    if(!check) printf( "Problems with quadB \n");

		    for( i1 = 0; i1 < neqel8; ++i1 )
		    {
			*(DB+i1) = *(Bh+i1)*D11+
				*(Bh+neqel8*1+i1)*D12;
			*(DB+neqel8*1+i1) = *(Bh+i1)*D21+
				*(Bh+neqel8*1+i1)*D22;
			*(DB+neqel8*2+i1) = *(Bh+neqel8*2+i1)*Gt;
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

	return 1;
}


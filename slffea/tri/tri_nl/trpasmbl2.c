/*
    This utility function assembles the P_global matrix for a finite
    element program which does analysis on a triangle which can behave
    nonlinearly.  It can also do assembly for the P_global_CG matrix
    if non-linear analysis using the conjugate gradient method is chosen.

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
#include "../tri/trconst.h"
#include "../tri/trstruct.h"

#define SMALL      1.e-20

extern int dof, numel, numnp, plane_stress_flag, sof, flag_3D,
	local_vec_flag;
extern double shg[sosh], shgh[sosh], shl[sosh], w[num_int], *Area0;
extern int Passemble_flag, Passemble_CG_flag;


int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int triangleB(double *,double *);

int dotX(double *, double *, double *, int);

int local_vectors( int *, double *, double * );

int tr2Passemble( int *connect, double *coord, double *coordh, int *el_matl,
	double *local_xyz, double *localh_xyz, MATL *matl, double *P_global,
	SDIM *stress, double *dU, double *P_global_CG, double *U)
{
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness;
	double fdum, fdum2, area_el, areah_el;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB],  Bh[soB], stressd[sdim];
	double DB[soB], K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq36], U_el[neqel];
	double rotate[nsd2*nsd], rotateh[nsd2*nsd];
	double P_el[neqel], P_el_local[neqel6], P_temp[neqel];
	double dU_el[neqel], dU_el_local[npel*nsd2], domega_el[1];
	double stress_el[sdim], dstress_el[sdim], dstrain_el[sdim];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double coordh_el[npel*nsd], coordh_el_trans[npel*nsd],
		coordh_el_local[npel*nsd2], coordh_el_local_trans[npel*nsd2];
	double det[num_int], deth[num_int];
	double X1, X2, X3, Y1, Y2, Y3;
	double X1h, X2h, X3h, Y1h, Y2h, Y3h;

	memset(P_global,0,dof*sof);
	memset(P_global_CG,0,dof*sof);

	if(flag_3D)
	{
	    check = local_vectors_triangle( connect, coord, local_xyz );
	    if(!check) printf( " Problems with local_vectors \n");

	    check = local_vectors_triangle( connect, coordh, localh_xyz );
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

/* Create the coord_el vector for one element */

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

		memset(rotate,0,nsd2*nsd*sof);
		memset(rotateh,0,nsd2*nsd*sof);

		if(!flag_3D)
		{
/* For 2-D meshes */
		    X1 = *(coord_el_trans);
		    X2 = *(coord_el_trans + 1);
		    X3 = *(coord_el_trans + 2);

		    Y1 = *(coord_el_trans + npel*1);
		    Y2 = *(coord_el_trans + npel*1 + 1);
		    Y3 = *(coord_el_trans + npel*1 + 2);

		    X1h = *(coordh_el_trans);
		    X2h = *(coordh_el_trans + 1);
		    X3h = *(coordh_el_trans + 2);

		    Y1h = *(coordh_el_trans + npel*1);
		    Y2h = *(coordh_el_trans + npel*1 + 1);
		    Y3h = *(coordh_el_trans + npel*1 + 2);

		    *(dU_el_local) = *(dU_el);
		    *(dU_el_local + 1) = *(dU_el + 1);

		    *(dU_el_local + 2) = *(dU_el + 3);
		    *(dU_el_local + 3) = *(dU_el + 4);

		    *(dU_el_local + 4) = *(dU_el + 6);
		    *(dU_el_local + 5) = *(dU_el + 7);

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

		    *(rotateh)     = *(localh_xyz + nsdsq*k);
		    *(rotateh + 1) = *(localh_xyz + nsdsq*k + 1);
		    *(rotateh + 2) = *(localh_xyz + nsdsq*k + 2);
		    *(rotateh + 3) = *(localh_xyz + nsdsq*k + 3);
		    *(rotateh + 4) = *(localh_xyz + nsdsq*k + 4);
		    *(rotateh + 5) = *(localh_xyz + nsdsq*k + 5);

/* Put coord_el and coordh_el into local coordinates */

		    dum = nsd*npel;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with rotXmat2 \n");
		    check = rotXmat2(coordh_el_local, rotateh, coordh_el, 1, dum);
		    if(!check) printf( "Problems with rotXmat2 \n");

/* Assembly of the B and Bh matrix.  This is taken from "Fundamentals of the Finite
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

		    X1h = *(coordh_el_local);     Y1h = *(coordh_el_local + 1);
		    X2h = *(coordh_el_local + 2); Y2h = *(coordh_el_local + 3);
		    X3h = *(coordh_el_local + 4); Y3h = *(coordh_el_local + 5);

/* Put dU_el into local coordinates */

		    check = rotXmat2(dU_el_local, rotate, dU_el, 1, nsd*npel);
		    if(!check) printf( "Problems with rotXmat2 \n");

		}

		*(coord_el_local_trans) = X1;     *(coord_el_local_trans + 3) = Y1;
		*(coord_el_local_trans + 1) = X2; *(coord_el_local_trans + 4) = Y2;
		*(coord_el_local_trans + 2) = X3; *(coord_el_local_trans + 5) = Y3;

		*(coordh_el_local_trans) = X1h;     *(coordh_el_local_trans + 3) = Y1h;
		*(coordh_el_local_trans + 1) = X2h; *(coordh_el_local_trans + 4) = Y2h;
		*(coordh_el_local_trans + 2) = X3h; *(coordh_el_local_trans + 5) = Y3h;

/* Area is simply the the cross product of the sides connecting node 1 to node 0
   and node 2 to node 0, divided by 2.
*/

		fdum = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);

		fdum2 = (X2h - X1h)*(Y3h - Y1h) - (X3h - X1h)*(Y2h - Y1h);

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		area_el = .5*fdum;
		areah_el = .5*fdum2;

		memset(P_el,0,neqel*sof);
		memset(P_el_local,0,neqel6*sof);
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
		memset(K_local,0,neqlsq36*sof);
		memset(P_temp,0,neqel6*sof);

		if(Passemble_CG_flag)
		{
		    memset(DB,0,soB*sof);
		    memset(K_temp,0,neqlsq*sof);
		}

/* Assembly of the B matrix at full time - Assembly of the Bh matrix at 1/2 time */
	
		*(B)    = Y2 - Y3;         *(Bh)    = Y2h - Y3h;
		*(B+2)  = Y3 - Y1;         *(Bh+2)  = Y3h - Y1h;
		*(B+4)  = Y1 - Y2;         *(Bh+4)  = Y1h - Y2h;
		*(B+7)  = X3 - X2;         *(Bh+7)  = X3h - X2h;
		*(B+9)  = X1 - X3;         *(Bh+9)  = X1h - X3h;
		*(B+11) = X2 - X1;         *(Bh+11) = X2h - X1h;
		*(B+12) = X3 - X2;         *(Bh+12) = X3h - X2h;
		*(B+13) = Y2 - Y3;         *(Bh+13) = Y2h - Y3h;
		*(B+14) = X1 - X3;         *(Bh+14) = X1h - X3h;
		*(B+15) = Y3 - Y1;         *(Bh+15) = Y3h - Y1h;
		*(B+16) = X2 - X1;         *(Bh+16) = X2h - X1h;
		*(B+17) = Y1 - Y2;         *(Bh+17) = Y1h - Y2h;

/* Calculation of the incremental strain at 1/2 time */

		check=matX(dstrain_el,Bh,dU_el_local, sdim, 1, neqel6);
		if(!check) printf( "Problems with matX \n");

/* Assembly of domega_el(rotation) of the element at 1/2 time = Bh*dU_el_local */

/* First, Bh is modified slightly before the calculaiton is performed */

		for( i = 0; i < npel; ++i )
		{
		    i2      = ndof2*i+1;

		    *(Bh+neqel6*2+i2) *= -1.0;
		}

		check=dotX((domega_el),(Bh+2*neqel6), dU_el_local, neqel6);
		if(!check) printf( "Problems with matX \n");

/* Calculation of the incremental Yaumann constitutive rate change */

		*(dstress_el) = *(dstrain_el)*D11 + *(dstrain_el+1)*D12;
		*(dstress_el+1) = *(dstrain_el)*D21 + *(dstrain_el+1)*D22;
		*(dstress_el+2) = *(dstrain_el+2)*Gt;

/* Update the global stress matrix at half time */

		stress[k].xx += *(dstress_el) +
		    *(domega_el)*stress[k].xy;
		stress[k].yy +=  *(dstress_el+1) -
		    *(domega_el)*stress[k].xy;
		stress[k].xy += *(dstress_el+2) +
		    .5*(*(domega_el)*(stress[k].yy -
		    stress[k].xx));

/* Update the global stress matrix at full time */

		stress[k].xx += +
		    *(domega_el)*stress[k].xy;
		stress[k].yy += -
		    *(domega_el)*stress[k].xy;
		stress[k].xy += +
		    .5*(*(domega_el)*(stress[k].yy -
		    stress[k].xx));

		if(Passemble_flag)
		{

/* Calculation of the element stress matrix at full time */

		    *(stress_el) = stress[k].xx;
		    *(stress_el+1) = stress[k].yy;
		    *(stress_el+2) = stress[k].xy;

/* Assembly of the element local P matrix = (B transpose)*stress_el  */

		    check=matXT(P_temp, B, stress_el, neqel6, 1, sdim);
		    if(!check) printf( "Problems with matXT \n");

		    for( i1 = 0; i1 < neqel6; ++i1 )
		    {
			    /*printf("rrrrr %14.9f \n",*(P_temp+i1));*/
			    /*printf("rrrrr %16.8e %16.8e \n",
				    *(P_temp+i1), *(P_el_local+i1));*/
			    *(P_el_local+i1) += *(P_temp+i1)/( 4.0*area_el );
		    }

/* Put the local P element matix into global coordinates.  */

		    if(!flag_3D)
		    {
/* For 2-D meshes */
			*(P_el) = *(P_el_local);
			*(P_el + 1) = *(P_el_local + 1);

			*(P_el + 3) = *(P_el_local + 2);
			*(P_el + 4) = *(P_el_local + 3);

			*(P_el + 6) = *(P_el_local + 4);
			*(P_el + 7) = *(P_el_local + 5);
		    }
		    else
		    {
			check = rotTXmat2(P_el, rotate, P_el_local, 1, neqel);
			if(!check) printf( "Problems with rotTXmat2 \n");
		    }
		}

		if(Passemble_CG_flag)
		{
/*
    Note that I do not use Bh for DB when the conjugate gradient
    method is used.  This is because I want to keep things consistent with
    trConjPassemble.
*/
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

		    for( j = 0; j < neqlsq36; ++j )
		    {
			*(K_el + j) = *(K_local + j)/(4.0*area_el);
		    }
		}

		if(Passemble_CG_flag)
		{
		    if(!flag_3D)
		    {
/*     For 2-D meshes */
			for( i = 0; i < npel; ++i )
			{
			    for( j = 0; j < npel; ++j )
			    {
/*     row for displacement x */
				*(K_temp + ndof*neqel*i + ndof*j) =
				    *(K_el + ndof2*neqel6*i + ndof2*j);
				*(K_temp + ndof*neqel*i + ndof*j + 1) =
				    *(K_el + ndof2*neqel6*i + ndof2*j + 1);
				*(K_temp + ndof*neqel*i + ndof*j + 2) = 0.0;

/*     row for displacement y */
				*(K_temp + ndof*neqel*i + neqel  + ndof*j) =
				    *(K_el + ndof2*neqel6*i + neqel6 + ndof2*j);
				*(K_temp + ndof*neqel*i + neqel + ndof*j + 1) =
				    *(K_el + ndof2*neqel6*i + neqel6 + ndof2*j + 1);
				*(K_temp + ndof*neqel*i + neqel + ndof*j + 2) = 0.0;

/*     row for displacement z */
				*(K_temp + ndof*neqel*i + 2*neqel + ndof*j) = 0.0;
				*(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 1) = 0.0;
				*(K_temp + ndof*neqel*i + 2*neqel + ndof*j + 2) = 0.0;
			    }
			}
			memcpy(K_el, K_temp, neqlsq*sizeof(double));
		    }
		    else
		    {
/*     For 3-D meshes */

/*     Put K back to global coordinates */

			check = matXrot2(K_temp, K_el, rotate, neqel6, neqel);
			if(!check) printf( "Problems with matXrot2 \n");

			check = rotTXmat2(K_el, rotate, K_temp, neqel, neqel);
			if(!check) printf( "Problems with rotTXmat2 \n");
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



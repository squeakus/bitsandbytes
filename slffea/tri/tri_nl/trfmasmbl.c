/*
    This utility function assembles the Mass and force matrix for a finite
    element program which does analysis on a triangle which can behave
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

    Note that I do not use Bh for DB in tr2Passemble when the conjugate
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
#include "../tri/trconst.h"
#include "../tri/trstruct.h"

#define  DEBUG         0

extern int numel, numnp, dof, sof, plane_stress_flag, flag_3D, local_vec_flag;
extern double shg[sosh], shg_node[sosh], shl[sosh], shl_node[sosh],
	shl_node2[sosh_node2], w[num_int], *Area0;

int matXrot2(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat2(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int triangleB( double *,double *);

int trshg( double *, int, int, double *, double *, double *);

int local_vectors( int *, double *, double * );

int trFMassemble( int *connect, double *coord, double *coordh, int *el_matl,
	double *force, double *mass, double *local_xyz, double *localh_xyz,
	MATL *matl, double *U, double *Area0)
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G, Gt, thickness;
	double D11,D12,D21,D22;
	double lamda, mu;
	double B[soB], Bh[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq], K_local[neqlsq36];
	double rotate[nsd2*nsd], rotateh[nsd2*nsd];
	double force_el[neqel], U_el[neqel], U_el_local[npel*(ndof-1)];
	double coord_el[npel*nsd], coord_el_trans[npel*nsd],
		coord_el_local[npel*nsd2], coord_el_local_trans[npel*nsd2];
	double coordh_el[npel*nsd], coordh_el_trans[npel*nsd],
		coordh_el_local[npel*nsd2], coordh_el_local_trans[npel*nsd2];
	double X1, X2, X3, Y1, Y2, Y3;
	double X1h, X2h, X3h, Y1h, Y2h, Y3h;
	double det[1], wXdet;
	double fdum;
	double mass_el[neqel];

/* initialize all variables  */
	memset(mass,0,dof*sof);

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

/* Put coord_el into local coordinates */

		    dum = nsd*npel;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");

		    check = rotXmat2(coordh_el_local, rotateh, coordh_el, 1, dum);
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

		    X1h = *(coordh_el_local);     Y1h = *(coordh_el_local + 1);
		    X2h = *(coordh_el_local + 2); Y2h = *(coordh_el_local + 3);
		    X3h = *(coordh_el_local + 4); Y3h = *(coordh_el_local + 5);
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

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 2*Area:%f\n", k, fdum);
			return 0;
		}

/* A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/

		*(Area0 + k) = .5*fdum;

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);
		memset(force_el,0,neqel*sof);
		memset(K_temp,0,neqlsq*sof);

		memset(B,0,soB*sof);
		memset(Bh,0,soB*sof);
		memset(DB,0,soB*sof);
		memset(K_local,0,neqlsq36*sof);
#if !DEBUG

/* For [B] below, see "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204.  Despite the permutation of nodes mentioned above, 
   B remains the same.  */

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
#endif

#if DEBUG

/* The code below is for debugging the code.  It uses shape functions
   to calculate the B matrix.  Normally, we loop through the number
   of integration points (num_int), but for triangles, the shape
   function derivatives are constant.
*/

/* at 1/2 time */
		check=trshg(det, k, 2, shl, shg, coord_el_local_trans);
		if(!check) printf( "Problems with trshg \n");

		check = triangleB(shg, B);
		if(!check) printf( "Problems with triangleB \n");

/* at full time */
		check=trshg(deth, k, 2, shl, shgh, coordh_el_local_trans);
		if(!check) printf( "Problems with trshg \n");

		check = triangleB(shgh, Bh);
		if(!check) printf( "Problems with triangleB \n");
#endif

		for( i1 = 0; i1 < neqel6; ++i1 )
		{
			*(DB+i1) = *(Bh+i1)*D11+
				*(Bh+neqel6*1+i1)*D12;
			*(DB+neqel6*1+i1) = *(Bh+i1)*D21+
				*(Bh+neqel6*1+i1)*D22;
			*(DB+neqel6*2+i1) = *(Bh+neqel6*2+i1)*Gt;
		}

		check=matXT(K_local, B, DB, neqel6, neqel6, sdim);
		if(!check) printf( "Problems with matXT  \n");

#if !DEBUG
		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_el + j) = *(K_local + j)/( 4.0*(*(Area0 + k)) );
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


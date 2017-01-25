/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a plate.  It is for modal analysis.

		Updated 11/4/06

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

extern int numel, numnp, dof, sof, flag_3D;
extern SH shg, shg_node, shl, shl_node;
extern double shl_node2[sosh_node2], w[num_int+1], *Area0;
extern int consistent_mass_flag, consistent_mass_store;

int matXrot(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int rotTXmat(double *, double *, double *, int, int);

int matXT(double *, double *, double *, int, int, int);

int plateBtr_mass(double *,double *);

int triangleB_mass(double *,double *);

int matX(double *, double *, double *, int, int, int);

int plMassPassemble_triangle(int *connect, double *coord, int *el_matl, double *local_xyz,
	double *mass, MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, dof_el6[neqel18], sdof_el[npel3*nsd];
	int check, node, counter, dum;
	int matl_num;
	double rho, thickness, thickness_cubed, fdum, fdum2;
	double B_mass[MsoBtr], B2_mass[MsoBtr], Bmem_mass[MsoBmemtr], B2mem_mass[MsoBmemtr];
	double M_temp[neqlsq324], M_el[neqlsq324], M_local[neqlsq81];
	double M_bend[neqlsq81], M_mem[neqlsq36];
	double rotate[nsdsq];
	double coord_el[npel3*nsd], coord_el_trans[npel3*nsd],
		coord_el_local[npel3*nsd2], coord_el_local_trans[npel3*nsd2];
	double X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3, X12, X13, Y12, Y13, Z12, Z13;
	double U_el[neqel18];
	double det[num_int3], area_el, wXdet;
	double P_el[neqel18];

	memset(P_global,0,dof*sof);

	memcpy(shg.bend,shl.bend,soshb*sizeof(double));

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel3; ++j )
		{
			node = *(connect+npel3*k+j);

			*(dof_el6 + ndof6*j) = ndof6*node;
			*(dof_el6 + ndof6*j + 1) = ndof6*node + 1;
			*(dof_el6 + ndof6*j + 2) = ndof6*node + 2;
			*(dof_el6 + ndof6*j + 3) = ndof6*node + 3;
			*(dof_el6 + ndof6*j + 4) = ndof6*node + 4;
			*(dof_el6 + ndof6*j + 5) = ndof6*node + 5;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, (mass + k*neqlsq324), U_el, neqel18, 1, neqel18);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel18; ++j )
		{
			*(P_global+*(dof_el6+j)) += *(P_el+j);
		}
	    }
	}
	else
	{

/* Assemble P matrix by re-deriving element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {
		matl_num = *(el_matl+k);
		thickness = matl[matl_num].thick;
		thickness_cubed = thickness*thickness*thickness/12.0;
		rho = matl[matl_num].rho;
		area_el = 0.0;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel3; ++j )
		{
			node = *(connect+npel3*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el+nsd*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+nsd*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+nsd*j+2)=*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel3*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel3*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(dof_el6+ndof6*j) = ndof6*node;
			*(dof_el6+ndof6*j+1) = ndof6*node+1;
			*(dof_el6+ndof6*j+2) = ndof6*node+2;
			*(dof_el6+ndof6*j+3) = ndof6*node+3;
			*(dof_el6+ndof6*j+4) = ndof6*node+4;
			*(dof_el6+ndof6*j+5) = ndof6*node+5;
		}

		memset(rotate,0,nsdsq*sof);

		if(!flag_3D)
		{
/* For 2-D meshes */
		    X1 = *(coord_el_trans);
		    X2 = *(coord_el_trans + 1);
		    X3 = *(coord_el_trans + 2);

		    Y1 = *(coord_el_trans + npel3*1);
		    Y2 = *(coord_el_trans + npel3*1 + 1);
		    Y3 = *(coord_el_trans + npel3*1 + 2);
		}
		else
		{

/* Th rotation matrix, rotate[], is used to rotate the global U_el, made up of
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
		    dum = nsd*npel3;
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

		area_el = .5*fdum;

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq324*sof);
		memset(M_temp,0,neqlsq324*sof);
		memset(M_bend,0,neqlsq81*sof);
		memset(M_mem,0,neqlsq36*sof);

/* FOR THE MEMBRANE */

/*
   This is taken from "Theory of Matrix Structural Analysis" by
   J. S. Przemieniecki, page 297-299.
*/

		fdum = area_el*rho*thickness/12.0;

		*(M_mem)    = 2.0*fdum; *(M_mem+2)  = 1.0*fdum; *(M_mem+4)  = 1.0*fdum;
		*(M_mem+7)  = 2.0*fdum; *(M_mem+9)  = 1.0*fdum; *(M_mem+11) = 1.0*fdum;
		*(M_mem+12) = 1.0*fdum; *(M_mem+14) = 2.0*fdum; *(M_mem+16) = 1.0*fdum;
		*(M_mem+19) = 1.0*fdum; *(M_mem+21) = 2.0*fdum; *(M_mem+23) = 1.0*fdum;
		*(M_mem+24) = 1.0*fdum; *(M_mem+26) = 1.0*fdum; *(M_mem+28) = 2.0*fdum;
		*(M_mem+31) = 1.0*fdum; *(M_mem+33) = 1.0*fdum; *(M_mem+35) = 2.0*fdum;

/* FOR THE BENDING */

/* The mass matrix below was derived by inspection by looking the M_bend which came from
   integrating the product of B_mass in the DEBUG code.
*/

		fdum = area_el*rho*thickness/6.0;
		fdum2 = area_el*rho*thickness_cubed/6.0;

		*(M_bend)    = 2.0*fdum;  *(M_bend+3)  = 1.0*fdum;  *(M_bend+6)  = 1.0*fdum;
		*(M_bend+10) = 2.0*fdum2; *(M_bend+13) = 1.0*fdum2; *(M_bend+16) = 1.0*fdum2;
		*(M_bend+20) = 2.0*fdum2; *(M_bend+23) = 1.0*fdum2; *(M_bend+26) = 1.0*fdum2;
		*(M_bend+27) = 1.0*fdum;  *(M_bend+30) = 2.0*fdum;  *(M_bend+33) = 1.0*fdum;
		*(M_bend+37) = 1.0*fdum2; *(M_bend+40) = 2.0*fdum2; *(M_bend+43) = 1.0*fdum2;
		*(M_bend+47) = 1.0*fdum2; *(M_bend+50) = 2.0*fdum2; *(M_bend+53) = 1.0*fdum2;
		*(M_bend+54) = 1.0*fdum;  *(M_bend+57) = 1.0*fdum;  *(M_bend+60) = 2.0*fdum;
		*(M_bend+64) = 1.0*fdum2; *(M_bend+67) = 1.0*fdum2; *(M_bend+70) = 2.0*fdum2;
		*(M_bend+74) = 1.0*fdum2; *(M_bend+77) = 1.0*fdum2; *(M_bend+80) = 2.0*fdum2;

		/* printf("This is Area %10.6f for element %4d\n",area_el,k);*/

/* Unlike the quadrilateral and triangle, the lines below are executed for both 2-D
   and 3-D elements because I wanted to make the rotation easier to understand.  This
   element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out M_bend to test if I can get the same results
   as the triangle element which has no bending.  Also, I need to zero out K_bend in
   "plkasmbl.c" with:

                memset(K_bend,0,neqlsq81*sof);
*/
#if 0
		memset(M_bend,0,neqlsq81*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel3; ++i )
		{
			for( j = 0; j < npel3; ++j )
			{
/* row for displacement x */
			   *(M_temp + ndof6*neqel18*i + ndof6*j) =
				*(M_mem + ndof2*neqel6*i + ndof2*j);
			   *(M_temp + ndof6*neqel18*i + ndof6*j + 1) =
				*(M_mem + ndof2*neqel6*i + ndof2*j + 1);
			   *(M_temp + ndof6*neqel18*i + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel18*i + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel18*i + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel18*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j) =
				*(M_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j);
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 1) =
				*(M_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j + 1);
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel9*i + ndof3*j);
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel9*i + ndof3*j + 1);
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel9*i + ndof3*j + 2);
			   *(M_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j);
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 1);
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 2);
			   *(M_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j);
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 1);
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 2);
			   *(M_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 5) = 0.0;
			}
		}
		memcpy(M_el, M_temp, neqlsq324*sizeof(double));

		if(flag_3D)
		{
/* For 3-D meshes */

/* Put M back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

		   check = matXrot(M_temp, M_el, rotate, neqel18, neqel18);
		   if(!check) printf( "Problems with matXrot \n");

		   check = rotTXmat(M_el, rotate, M_temp, neqel18, neqel18);
		   if(!check) printf( "Problems with rotTXmat \n");
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, M_el, U_el, neqel18, 1, neqel18);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel18; ++j )
		{
			*(P_global+*(dof_el6+j)) += *(P_el+j);
		}

	    }
	}

	return 1;
}

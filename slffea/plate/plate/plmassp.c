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

int matX(double *, double *, double *, int, int, int);

int plateB_mass(double *,double *);

int plshg_mass( double *, int, SH , double *);

int plMassPassemble(int *connect, double *coord, int *el_matl, double *local_xyz,
	double *mass, MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, j, k, dof_el6[neqel24], sdof_el[npel4*nsd];
	int check, node, counter, dum;
	int matl_num;
	double rho, thickness, thickness_cubed, fdum, fdum2;
	double B_mass[MsoB], B2_mass[MsoB], Bmem_mass[MsoBmem], B2mem_mass[MsoBmem];
	double M_temp[neqlsq576], M_el[neqlsq576], M_local[neqlsq144];
	double M_bend[neqlsq144], M_mem[neqlsq64];
	double rotate[nsdsq];
	double U_el[neqel24];
	double coord_el[npel4*nsd], coord_el_trans[npel4*nsd],
		coord_el_local[npel4*nsd2], coord_el_local_trans[npel4*nsd2];
	double det[num_int4], area_el, wXdet;
	double P_el[neqel24];

	memset(P_global,0,dof*sof);

	memcpy(shg.bend,shl.bend,soshb*sizeof(double));

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel4; ++j )
		{
			node = *(connect+npel4*k+j);

			*(dof_el6 + ndof6*j) = ndof6*node;
			*(dof_el6 + ndof6*j + 1) = ndof6*node + 1;
			*(dof_el6 + ndof6*j + 2) = ndof6*node + 2;
			*(dof_el6 + ndof6*j + 3) = ndof6*node + 3;
			*(dof_el6 + ndof6*j + 4) = ndof6*node + 4;
			*(dof_el6 + ndof6*j + 5) = ndof6*node + 5;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, (mass + k*neqlsq576), U_el, neqel24, 1, neqel24);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel24; ++j )
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
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_trans + j);
			*(coord_el_local_trans + 1*npel4 + j) = *(coord_el_trans + npel4*1 + j);
		    }
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
		    dum = nsd*npel4;
		    check = rotXmat2(coord_el_local, rotate, coord_el, 1, dum);
		    if(!check) printf( "Problems with  rotXmat2 \n");
		    for( j = 0; j < npel4; ++j )
		    {
			*(coord_el_local_trans + j) = *(coord_el_local+nsd2*j);
			*(coord_el_local_trans + npel4*1 + j) = *(coord_el_local+nsd2*j+1);
		    }
		}

/* The call to plshg_mass is only for calculating the determinent */

		check = plshg_mass(det, k, shg, coord_el_local_trans);
		if(!check) printf( "Problems with plshg_mass \n");

/* The loop over j below calculates the 4 points of the gaussian integration
   for several quantities */

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq576*sof);
		memset(M_temp,0,neqlsq576*sof);
		memset(M_bend,0,neqlsq144*sof);
		memset(M_mem,0,neqlsq64*sof);

		for( j = 0; j < num_int; ++j )
		{

/* FOR THE MEMBRANE */

		    memset(Bmem_mass,0,MsoBmem*sof);
		    memset(B2mem_mass,0,MsoBmem*sof);
		    memset(M_local,0,neqlsq64*sof);

/* Assembly of the B matrix for mass */

		    check = quadB_mass((shg.bend+npel4*(nsd2+1)*j + npel4*(nsd2)),Bmem_mass);
		    if(!check) printf( "Problems with quadB_mass \n");

		    memcpy(B2mem_mass,Bmem_mass,MsoBmem*sizeof(double));

		    wXdet = *(w+j)*(*(det+j));

/* Calculate the Area from determinant of the Jacobian */

		    area_el += wXdet;

		    check=matXT(M_local, Bmem_mass, B2mem_mass, neqel8, neqel8, nsd2);
		    if(!check) printf( "Problems with matXT \n");

/* Normally, with plane elements, we assume a unit thickness in the transverse direction.  But
   because these elements can be 3 dimensional, I multiply the material property matrix by the
   thickness.  This is justified by equation 5.141a in "Theory of Matrix Structural
   Analysis" by J. S. Przemieniecki on page 86.
*/

		    fdum = rho*thickness*wXdet;
		    for( i2 = 0; i2 < neqlsq64; ++i2 )
		    {
			*(M_mem+i2) += *(M_local+i2)*fdum;
		    }

/* FOR THE BENDING */

		    memset(B_mass,0,MsoB*sof);
		    memset(B2_mass,0,MsoB*sof);
		    memset(M_local,0,neqlsq144*sof);

/* Assembly of the B matrix for mass */

		    check = plateB_mass((shg.bend+npel4*(nsd2+1)*j + npel4*(nsd2)),B_mass);
		    if(!check) printf( "Problems with plateB_mass \n");

		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

		    wXdet = *(w+j)*(*(det+j));

		    check=matXT(M_local, B_mass, B2_mass, neqel12, neqel12, nsd2+1);
		    if(!check) printf( "Problems with matXT \n");

		    fdum = rho*thickness*wXdet;
		    fdum2 = rho*thickness_cubed*wXdet;

		    for( i2 = 0; i2 < npel4; ++i2 )
		    {
			*(M_bend+ndof3*i2) +=
			    *(M_local+ndof3*i2)*fdum;
			*(M_bend+neqel12*1+ndof3*i2+1) +=
			    *(M_local+neqel12*1+ndof3*i2+1)*fdum2;
			*(M_bend+neqel12*2+ndof3*i2+2) +=
			    *(M_local+neqel12*2+ndof3*i2+2)*fdum2;

			*(M_bend+neqel12*3+ndof3*i2) +=
			    *(M_local+neqel12*3+ndof3*i2)*fdum;
			*(M_bend+neqel12*4+ndof3*i2+1) +=
			    *(M_local+neqel12*4+ndof3*i2+1)*fdum2;
			*(M_bend+neqel12*5+ndof3*i2+2) +=
			    *(M_local+neqel12*5+ndof3*i2+2)*fdum2;

			*(M_bend+neqel12*6+ndof3*i2) +=
			    *(M_local+neqel12*6+ndof3*i2)*fdum;
			*(M_bend+neqel12*7+ndof3*i2+1) +=
			    *(M_local+neqel12*7+ndof3*i2+1)*fdum2;
			*(M_bend+neqel12*8+ndof3*i2+2) +=
			    *(M_local+neqel12*8+ndof3*i2+2)*fdum2;

			*(M_bend+neqel12*9+ndof3*i2) +=
			    *(M_local+neqel12*9+ndof3*i2)*fdum;
			*(M_bend+neqel12*10+ndof3*i2+1) +=
			    *(M_local+neqel12*10+ndof3*i2+1)*fdum2;
			*(M_bend+neqel12*11+ndof3*i2+2) +=
			    *(M_local+neqel12*11+ndof3*i2+2)*fdum2;
		    }
		}

		/* printf("This is Area %10.6f for element %4d\n",area_el,k);*/

/* Unlike the quadrilateral and triangle element, the lines below are executed for both 2-D
   and 3-D elements because I wanted to make the rotation easier to understand.  This
   element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out M_bend to test if I can get the same results
   as the quadrilateral element which has no bending.  Also, I need to zero out K_bend in
   "plkasmbl.c" with:

                memset(K_bend,0,neqlsq144*sof);

   Note that if an eigenmode comparison is made, I must change the num_eigen in "fempl.c" to:

                num_eigen = (int)(2.0*nmode);

   to match the quad. 
*/
#if 0
		memset(M_bend,0,neqlsq144*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel4; ++i )
		{
			for( j = 0; j < npel4; ++j )
			{
/* row for displacement x */
			   *(M_temp + ndof6*neqel24*i + ndof6*j) =
				*(M_mem + ndof2*neqel8*i + ndof2*j);
			   *(M_temp + ndof6*neqel24*i + ndof6*j + 1) =
				*(M_mem + ndof2*neqel8*i + ndof2*j + 1);
			   *(M_temp + ndof6*neqel24*i + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel24*i + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel24*i + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel24*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j) =
				*(M_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j);
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 1) =
				*(M_mem + ndof2*neqel8*i + 1*neqel8 + ndof2*j + 1);
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 1*neqel24 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel12*i + ndof3*j);
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel12*i + ndof3*j + 1);
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel12*i + ndof3*j + 2);
			   *(M_temp + ndof6*neqel24*i + 2*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j);
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 1);
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel12*i + 1*neqel12 + ndof3*j + 2);
			   *(M_temp + ndof6*neqel24*i + 3*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 2) =
				*(M_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j);
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 3) =
				*(M_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 1);
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 4) =
				*(M_bend + ndof3*neqel12*i + 2*neqel12 + ndof3*j + 2);
			   *(M_temp + ndof6*neqel24*i + 4*neqel24 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 1) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 2) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 3) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 4) = 0.0;
			   *(M_temp + ndof6*neqel24*i + 5*neqel24 + ndof6*j + 5) = 0.0;
			}
		}
		memcpy(M_el, M_temp, neqlsq576*sizeof(double));

		if(flag_3D)
		{
/* For 3-D meshes */

/* Put M back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

		   check = matXrot(M_temp, M_el, rotate, neqel24, neqel24);
		   if(!check) printf( "Problems with matXrot \n");

		   check = rotTXmat(M_el, rotate, M_temp, neqel24, neqel24);
		   if(!check) printf( "Problems with rotTXmat \n");
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel24; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(P_el, M_el, U_el, neqel24, 1, neqel24);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel24; ++j )
		{
			*(P_global+*(dof_el6+j)) += *(P_el+j);
		}

	    }
	}

	return 1;
}

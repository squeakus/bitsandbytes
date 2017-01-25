/*
    This utility function assembles the K matrix for a finite 
    element program which does analysis on plate elements made
    up of 3 node triangles.

    There are several things to say about this 3 node plate element.
    When I originally intended to implement it, I wanted to go with
    what was prescribed in:

       Przemieniecki, J. S., Theory of Matrix Structural Analysis,
           Dover Publications Inc., New York, 1985.

    page 109-115.  But his element, in addition to the complex
    interpolation given for the displacements, required a matrix
    inversion of the 9X9 matrix given by Equation 5.238 on page 112 for
    each element.  Since this matrix has many zeros, I considered
    working an explicit expression for it and may do so later, but
    for now, it seems too much to do.   So I based my element on
    the 4 node plate in terms of [B] and [D], but I used the linear
    interpoltating shape functions for the triangle given in the
    triangle element.  I also looked at the ANSYS manual:

       Kohneke, Peter, *ANSYS User's Manual for Revision 5.0,
           Vol IV Theory, Swanson Analysis Systems Inc., Houston, 1992.

    and it seems to do the same as I did as can be seen by looking on
    pages 14-156 to 14-158 for SHELL43.  Also, see pages 12-10 to
    page 12-12, particularly Section 12.5.4 for the 3-D 3-node
    Triangular Shells With RDOF and With SD.  These equations closely
    express what I have done with this plate albeit without the
    membrane components.

    One thing to note is that the material property matrix [D] as
    given on page 14-158 in equation 14.43-1 is different for that
    of the plate that I am using below.

    This element has problems related to shear locking.  This is mentioned
    in:

        Benson, David, AMES 232 B Course Notes: Winter Quarter 1997,  UCSD
            Soft Reserves, La Jolla, 1997.

    page 31.  Benson says the cause is that it is "hard to interpolate constant
    curvature modes on a triange".  His recommendation is to use the
    Hughes-Tezduyar element which interpolates the shear from the midpoint
    of each side along with other modifications as well.

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

#define  DEBUG         0

extern int analysis_flag, dof, neqn, numel, numnp, sof, plane_stress_flag, flag_3D;
extern int gauss_moment_flag;
extern int LU_decomp_flag, numel_K, numel_P;
extern double wtr[num_int3], *Area0;
extern double shgtr[soshtr], shgtr_node[soshtr], shltr[soshtr], shltr_node[soshtr],
	shltr_node2[soshtr_node2];

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int rotTXmat(double *, double *, double *, int, int);

int rotXmat2(double *, double *, double *, int, int);

int matX( double *,double *,double *, int ,int ,int );

int matXT( double *, double *, double *, int, int, int);

int plateBtr( double *, double *);

int plateB1pt( double *, double *);

int triangleB( double *,double *);

int trshg( double *, int, int, double *, double *, double *);

int plKassemble_triangle(double *A, int *connect, double *coord, CURVATURE *curve, MDIM *curve_node,
	int *el_matl, double *force, int *id, int *idiag, double *K_diag, int *lm, double *local_xyz,
	MATL *matl, MOMENT *moment, MDIM *moment_node, double *node_counter, STRAIN *strain,
	SDIM *strain_node, STRESS *stress, SDIM *stress_node, double *U, double *Uz_fib,
	double *Arean)
{
	int i, i1, i2, j, k, dof_el6[neqel18], dof_el[neqel15], sdof_el[npel3*nsd];
	int check, counter, node, dum;
	int matl_num;
	double Emod, Pois, G1, G2, G3, thickness, shearK, const1, const2;
	double D11,D12,D21,D22,D11mem,D12mem,D21mem,D22mem,Gmem;
	double lamda, mu;
	double B[soBtr], DB[soBtr], Bmem[soBmemtr], DBmem[soBmemtr];
	double K_temp[neqlsq324], K_el[neqlsq324], K_local[neqlsq81];
	double K_bend[neqlsq81], K_mem[neqlsq36];
	double rotate[nsdsq];
	double force_el[neqel18], U_el[neqel18], U_el_local[neqel18], U_el_mem_local[neqel6];
	double coord_el[npel3*nsd], coord_el_trans[npel3*nsd],
		coord_el_local[npel3*nsd], coord_el_local_trans[npel3*nsd];
	double X1, X2, X3, Y1, Y2, Y3, Z1, Z2, Z3, X12, X13, Y12, Y13, Z12, Z13;
	double stress_el[sdim5], strain_el[sdim5], xxaddyy, xxsubyy, xysq, invariant[nsd],
		yzsq, zxsq, xxyy;
	double det[1], wXdet;
	double fdum;

	for( k = 0; k < numel; ++k )
	{

		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;
		shearK = matl[matl_num].shear;

/* Due to the problem of shear locking, I temporarily divided the shear by 320.0
   but have now removed this.

		shearK = matl[matl_num].shear/320.0;
*/

		mu = Emod/(1.0+Pois)/2.0;

/* The lamda below is for plane strain */

		lamda = Emod*Pois/((1.0+Pois)*(1.0-2.0*Pois));

/* Recalculate lamda for plane stress */

		if(plane_stress_flag)
			lamda = Emod*Pois/(1.0-Pois*Pois);

/* FOR THE MEMBRANE */

		D11mem = thickness*(lamda+2.0*mu);
		D12mem = thickness*lamda;
		D21mem = thickness*lamda;
		D22mem = thickness*(lamda+2.0*mu);

		Gmem = thickness*mu;

/* FOR THE BENDING */

		const1 = thickness*thickness*thickness/12.0;
		const2 = thickness*shearK;

		/*printf("lamda, mu, Emod, Pois %f %f %f %f \n", lamda, mu, Emod, Pois);*/

		D11 = (lamda+2.0*mu)*const1;
		D12 = lamda*const1;
		D21 = lamda*const1;
		D22 = (lamda+2.0*mu)*const1;
		G1 = mu*const1;
		G2 = mu*const2;
		G3 = mu*const2;

		/*printf("lamda, mu, Emod, Pois  %f %f %f %f \n", lamda, mu, Emod, Pois);*/

/* Create the coord_el vector for one element */

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

		memset(U_el,0,neqel18*sof);
		memset(K_el,0,neqlsq81*sof);
		memset(force_el,0,neqel18*sof);
		memset(K_temp,0,neqlsq324*sof);
		memset(K_bend,0,neqlsq81*sof);
		memset(K_mem,0,neqlsq36*sof);
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
		    *(rotate + 6) = *(local_xyz + nsdsq*k + 6);
		    *(rotate + 7) = *(local_xyz + nsdsq*k + 7);
		    *(rotate + 8) = *(local_xyz + nsdsq*k + 8);

/* Put coord_el into local coordinates */

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

		*(Arean + k) = .5*fdum;

		memset(B,0,soBtr*sof);
		memset(DB,0,soBtr*sof);
		memset(Bmem,0,soBmemtr*sof);
		memset(DBmem,0,soBmemtr*sof);

		memset(K_local,0,neqlsq81*sof);


#if !DEBUG

/* FOR THE MEMBRANE */

/* For [Bmem] below, see "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204.  Despite the permutation of nodes mentioned above,
   B remains the same.  */

		*(Bmem) = Y2 - Y3;
		*(Bmem+2) = Y3 - Y1;
		*(Bmem+4) = Y1 - Y2;
		*(Bmem+7) = X3 - X2;
		*(Bmem+9) = X1 - X3;
		*(Bmem+11) = X2 - X1;
		*(Bmem+12) = X3 - X2;
		*(Bmem+13) = Y2 - Y3;
		*(Bmem+14) = X1 - X3;
		*(Bmem+15) = Y3 - Y1;
		*(Bmem+16) = X2 - X1;
		*(Bmem+17) = Y1 - Y2;


/* FOR THE BENDING */

/* See "Fundamentals of the Finite Element Method" by Hartley Grandin Jr.,
   page 205, Eq. 6.8 on page 204 for a description of how the differences in
   Xs and Ys translate into shape function derivatives.   As you can see, these
   derivatives are constant.

   Also, look on page 13-3 to 13-6 in the "ANSYS User's Manual".  It
   discusses integration points for triangles.  For 1 point integration,
   r = s = t = 1/3 and the weight is 1.

   Note that fdum divides all the terms on the RHS of the equations for [B],
   which means the [B] calculated below and the [B] from:

      check = plateBtr(shgtr, B);

   in the DEBUG case are the same.
*/

		fdum = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);
		*(Arean + k) = .5*fdum;

		*(B+2) = (Y3 - Y2)/fdum;
		*(B+5) = (Y1 - Y3)/fdum;
		*(B+8) = (Y2 - Y1)/fdum;
		*(B+10) = (X3 - X2)/fdum;
		*(B+13) = (X1 - X3)/fdum;
		*(B+16) = (X2 - X1)/fdum;
		*(B+19) = (Y2 - Y3)/fdum;
		*(B+20) = (X2 - X3)/fdum;
		*(B+22) = (Y3 - Y1)/fdum;
		*(B+23) = (X3 - X1)/fdum;
		*(B+25) = (Y1 - Y2)/fdum;
		*(B+26) = (X1 - X2)/fdum;

		*(B+27) = (Y2 - Y3)/fdum;
		*(B+29) = 1.0/3.0;
		*(B+30) = (Y3 - Y1)/fdum;
		*(B+32) = 1.0/3.0;
		*(B+33) = (Y1 - Y2)/fdum;
		*(B+35) = 1.0/3.0;

		*(B+36) = (X3 - X2)/fdum;
		*(B+37) = -1.0/3.0;
		*(B+39) = (X1 - X3)/fdum;
		*(B+40) = -1.0/3.0;
		*(B+42) = (X2 - X1)/fdum;
		*(B+43) = -1.0/3.0;
#endif

#if DEBUG

/* The code below is for debugging the code.  It uses shape functions
   to calculate the B matrix.  Normally, we loop through the number
   of integration points (num_int), but for triangles, the shape
   function derivatives are constant.
*/
		check=trshg(det, k, 2, shltr, shgtr, coord_el_local_trans);
		if(!check) printf( "Problems with trshg \n");

/* FOR THE MEMBRANE */

		check = triangleB(shgtr, Bmem);
		if(!check) printf( "Problems with triangleB \n");

/* FOR THE BENDING */

/* Calculate the lower part of the B and DB matrices for the shear */

		check = plateBtr(shgtr, B);
		if(!check) printf( "Problems with plateBtr \n");
#endif

/* FOR THE MEMBRANE */

		for( i1 = 0; i1 < neqel6; ++i1 )
		{
			*(DBmem+i1) = *(Bmem+i1)*D11mem+
				*(Bmem+neqel6*1+i1)*D12mem;
			*(DBmem+neqel6*1+i1) = *(Bmem+i1)*D21mem+
				*(Bmem+neqel6*1+i1)*D22mem;
			*(DBmem+neqel6*2+i1) = *(Bmem+neqel6*2+i1)*Gmem;
		}

		check=matXT(K_local, Bmem, DBmem, neqel6, neqel6, sdim3);
		if(!check) printf( "Problems with matXT  \n");

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
			*(K_mem + j) = *(K_local + j)*.5*(*(det));
		}
#endif
#if !DEBUG
		for( j = 0; j < neqlsq36; ++j )
		{
			*(K_mem + j) = *(K_local + j)/( 4.0*(*(Arean + k)) );
		}
#endif


/* FOR THE BENDING */

		for( i1 = 0; i1 < neqel9; ++i1 )
		{
			*(DB+i1) = *(B+i1)*D11 +
				*(B+neqel9*1+i1)*D12;
			*(DB+neqel9*1+i1) = *(B+i1)*D21 +
				*(B+neqel9*1+i1)*D22;
			*(DB+neqel9*2+i1) = *(B+neqel9*2+i1)*G1;
			*(DB+neqel9*3+i1) = *(B+neqel9*3+i1)*G2;
			*(DB+neqel9*4+i1) = *(B+neqel9*4+i1)*G3;
		}

		check=matXT(K_local, B, DB, neqel9, neqel9, sdim5);
		if(!check) printf( "Problems with matXT \n");
#if DEBUG

/* The code below is for debugging the code.  Normally, I would use
   .5*(*(det)) rather than wXdet,

   Originally, I used the 3 point rule, but because shear is under
   integrated using 1 point gauss, I switched back to the 1 point rule.

   For triangle elements, shape funciton derivatives are constant
   but the shape function itself changes based on the position of
   the interpolation variable in the isoparametric domain.   Because
   the [B] matrix for a plate, unlike a plane stress/strain triangle,
   has the shape functions themselves in addition to their derivatives
   which are constant, the choice of integration method matters. 
   The determinant though, is still constant because the shape funciton
   derivatives are constant so we only need *(det+0).

   Finally instead of using wXdet, for the 1 point rule, I use .5*(*(det))
   with w = 1.0.  
   
   A factor of 0.5 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174

   The !DEBUG and DEBUG methods of calculating K_bend are essentially the
   same with .5*(*(det)) = *(Arean + k), but since *(det) is only calculated
   in the DEBUG case, I leave the code below as is.
*/
		for( j = 0; j < neqlsq81; ++j )
		{
			*(K_bend + j) = *(K_local + j)*.5*(*(det));
		}

#endif

#if !DEBUG

		for( j = 0; j < neqlsq81; ++j )
		{
			/* *(K_bend + j) = *(K_local + j)/( 4.0*(*(Arean + k)) );*/
			*(K_bend + j) = *(K_local + j)*(*(Arean + k));
			/* *(K_bend + j) = *(K_local + j);*/
		}
#endif

/* Unlike the quadrilateral and triangle membrane elements, the lines below are executed
   for both 2-D and 3-D elements because I wanted to make the rotation easier to understand.
   This element is unique in that within the plane of the plate, there is only a z component
   of displacement along with rotations in x and y.  Because of this mismatch, it is more
   straightforward to create stiffnesses with components for x, y, and z displacements
   and rotations.  By doing this, I can take advantage of the rotation functions written
   for the beam.
*/

/* I am using the line below which zeros out K_bend to test if I can get the same results
   as the triangle element which has no bending.  Also, I need to zero out M_bend in
   "plmasmbl2.c" with:

                memset(M_bend,0,neqlsq81*sof);
*/
#if 0
		memset(K_bend,0,neqlsq81*sof);
#endif

/* For 2-D and 3-D meshes */
		for( i = 0; i < npel3; ++i )
		{
			for( j = 0; j < npel3; ++j )
			{
/* row for displacement x */
			   *(K_temp + ndof6*neqel18*i + ndof6*j) =
				*(K_mem + ndof2*neqel6*i + ndof2*j);
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 1) =
				*(K_mem + ndof2*neqel6*i + ndof2*j + 1);
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + ndof6*j + 5) = 0.0;

/* row for displacement y */
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j) =
				*(K_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j);
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 1) =
				*(K_mem + ndof2*neqel6*i + 1*neqel6 + ndof2*j + 1);
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 1*neqel18 + ndof6*j + 5) = 0.0;

/* row for displacement z */
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 2*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle x */
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + 1*neqel9 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 3*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle y */
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 2) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 3) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 1);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 4) =
				*(K_bend + ndof3*neqel9*i + 2*neqel9 + ndof3*j + 2);
			   *(K_temp + ndof6*neqel18*i + 4*neqel18 + ndof6*j + 5) = 0.0;

/* row for angle z */
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 1) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 2) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 3) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 4) = 0.0;
			   *(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*j + 5) = 0.0;
			}
/* The line below is an attempt to add an artificial stiffness to the rotation in local z.
   The triangle plate has much more problems than the quad plate for modal analysis, so I
   added this to improve results.
*/ 
			*(K_temp + ndof6*neqel18*i + 5*neqel18 + ndof6*i + 5) = 10.0e20;
		}
		memcpy(K_el, K_temp, neqlsq324*sizeof(double));

		/*for( i1 = 6; i1 < neqel18; ++i1 )
		{
		   for( i2 = 6; i2 < neqel18; ++i2 )
		   {
			printf("%10.5f ",*(K_el+neqel18*i1+i2));
		   }
		   printf("\n");
		}*/

		/*for( i1 = 0; i1 < neqel18; ++i1 )
		{
		   for( i2 = 0; i2 < neqel18; ++i2 )
		   {
			printf("%12.5e ",*(K_el+neqel18*i1+i2));  
		   }
		   printf("\n"); 
		}
		printf("aaaa\n");*/


		if(flag_3D)
		{
/* For 3-D meshes */

/* Put K back to global coordinates

   Note that I am using the same rotation functions written for the beam and truss.
*/

		   check = matXrot(K_temp, K_el, rotate, neqel18, neqel18);
		   if(!check) printf( "Problems with matXrot \n");

		/*for( i1 = 0; i1 < neqel18; ++i1 )
		{
		   for( i2 = 0; i2 < neqel18; ++i2 )
		   {
			printf("%12.5e ",*(K_temp+neqel18*i1+i2));
		   }
		   printf("\n");
		}
		printf("bbbb %d %d\n", neqel18, k);*/

		   check = rotTXmat(K_el, rotate, K_temp, neqel18, neqel18);
		   if(!check) printf( "Problems with rotTXmat \n");
		}


		/*for( i1 = 0; i1 < neqel18; ++i1 )
		{
		   for( i2 = 0; i2 < neqel18; ++i2 )
		   {
			printf("%12.5e ",*(K_el+neqel18*i1+i2));  
		   }
		   printf("\n"); 
		}
		printf("cccc %d %d\n", neqel18, k);*/


		for( j = 0; j < neqel18; ++j )
		{
			*(U_el + j) = *(U + *(dof_el6+j));
		}

		check = matX(force_el, K_el, U_el, neqel18, 1, neqel18);
		if(!check) printf( "Problems with matX \n");

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel18; ++j )
			{
				*(force + *(dof_el6+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(A, idiag, K_el, (lm + k*neqel18),
				neqel18);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(A, dof_el6, k, K_diag, K_el,
				neqel18, neqlsq324, numel_K);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel18; ++j )
			{
				*(force + *(dof_el6+j)) += *(force_el + j);
			}

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
			}
			else
			{
/* For 3-D meshes */

/* Put U_el into local coordinates */

#if 0
			   check = rotXmat2(U_el_local, rotate, U_el, 1, neqel18);
			   if(!check) printf( "Problems with rotXmat2 \n");
#endif
			   check = rotXmat(U_el_local, rotate, U_el, 1, neqel18);
			   if(!check) printf( "Problems with rotXmat4 \n");

/* FOR THE MEMBRANE */

			   *(U_el_mem_local) = *(U_el_local);
			   *(U_el_mem_local + 1) = *(U_el_local + 1);

			   *(U_el_mem_local + 2) = *(U_el_local + 6);
			   *(U_el_mem_local + 3) = *(U_el_local + 7);

			   *(U_el_mem_local + 4) = *(U_el_local + 12);
			   *(U_el_mem_local + 5) = *(U_el_local + 13);

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

/* Set the local z displacement vector to the Uz lamina vector */

			   *(Uz_fib + *(connect+npel3*k)) = *(U_el_local + 2);
			   *(Uz_fib + *(connect+npel3*k + 1)) = *(U_el_local + 8);
			   *(Uz_fib + *(connect+npel3*k + 2)) = *(U_el_local + 14);

			   memcpy(U_el_local, U_el, neqel18*sizeof(double));
			}

/* FOR THE MEMBRANE */

/* Calculation of the local strain matrix */

			check=matX(strain_el, Bmem, U_el_mem_local, sdim3, 1, neqel6 );
			if(!check) printf( "Problems with matX \n");
#if 0
			for( i1 = 0; i1 < sdim3; ++i1 )
			{
				 printf("%16.8e ",*(strain_el+i1));
				 /*printf("%16.8e ",*(strain_el+i1));
				 printf("%16.8e ",*(B+i1));*/
			}
			printf("%4d \n", k);
#endif

#if !DEBUG
			*(strain_el) /= 2.0*(*(Arean + k));
			*(strain_el + 1) /= 2.0*(*(Arean + k));
			*(strain_el + 2) /= 2.0*(*(Arean + k));
#endif

/* Update of the global strain matrix */

			strain[k].pt[0].xx = *(strain_el);
			strain[k].pt[0].yy = *(strain_el+1);
			strain[k].pt[0].xy = *(strain_el+2);

/* Calculation of the local stress matrix */

			*(stress_el) = strain[k].pt[0].xx*D11mem +
				strain[k].pt[0].yy*D12mem;
			*(stress_el+1) = strain[k].pt[0].xx*D21mem +
				strain[k].pt[0].yy*D22mem;
			*(stress_el+2) = strain[k].pt[0].xy*Gmem;

/* Update of the global stress matrix */

			stress[k].pt[0].xx += *(stress_el);
			stress[k].pt[0].yy += *(stress_el+1);
			stress[k].pt[0].xy += *(stress_el+2);

			for( j = 0; j < npel3; ++j )
			{
			    node = *(connect+npel3*k+j);

/* Add all the strains for a particular node from all the elements which share that node */

			    strain_node[node].xx += strain[k].pt[0].xx;
			    strain_node[node].yy += strain[k].pt[0].yy;
			    strain_node[node].xy += strain[k].pt[0].xy;

/* Add all the stresses for a particular node from all the elements which share that node */

			    stress_node[node].xx += stress[k].pt[0].xx;
			    stress_node[node].yy += stress[k].pt[0].yy;
			    stress_node[node].xy += stress[k].pt[0].xy;
			}

/* FOR THE BENDING */

/* Calculate the element moments.  Note that curve and moment are
   constant over a 3 node triangle element */

			memset(stress_el,0,sdim5*sof);
			memset(strain_el,0,sdim5*sof);

/* Calculation of the local curve matrix */


			check=matX(strain_el, B, U_el_local, sdim5, 1, neqel9 );
			if(!check) printf( "Problems with matX \n");

#if 0
			for( i1 = 0; i1 < sdim5; ++i1 )
			{
				 printf("%16.8e ",*(strain_el+i1));
				 /*printf("%16.8e ",*(strain_el+i1));
				 printf("%16.8e ",*(B+i1));*/
			}
			printf("%4d %16.8e\n", k, 2.0*(*(Arean + k)));
#endif

/* Update of the global strain matrix */

			curve[k].pt[0].xx = *(strain_el);
			curve[k].pt[0].yy = *(strain_el+1);
			curve[k].pt[0].xy = *(strain_el+2);
			strain[k].pt[0].zx = *(strain_el+3);
			strain[k].pt[0].yz = *(strain_el+4);

/* Calculate the principal straines */

			xxaddyy = .5*(curve[k].pt[0].xx + curve[k].pt[0].yy);
			xxsubyy = .5*(curve[k].pt[0].xx - curve[k].pt[0].yy);
			xysq = curve[k].pt[0].xy*curve[k].pt[0].xy;

			curve[k].pt[0].I = xxaddyy + sqrt( xxsubyy*xxsubyy
				+ xysq);
			curve[k].pt[0].II = xxaddyy - sqrt( xxsubyy*xxsubyy
				+ xysq);
			/*printf("%14.6e %14.6e %14.6e\n",xxaddyy,xxsubyy,xysq);*/

/* Calculation of the local stress matrix */

			*(stress_el)=curve[k].pt[0].xx*D11 +
				curve[k].pt[0].yy*D12;
			*(stress_el+1)=curve[k].pt[0].xx*D21 +
				curve[k].pt[0].yy*D22;
			*(stress_el+2)=curve[k].pt[0].xy*G1;
			*(stress_el+3)=strain[k].pt[0].zx*G2;
			*(stress_el+4)=strain[k].pt[0].yz*G3;

/* Update of the global stress matrix */

			moment[k].pt[0].xx += *(stress_el);
			moment[k].pt[0].yy += *(stress_el+1);
			moment[k].pt[0].xy += *(stress_el+2);
			stress[k].pt[0].zx += *(stress_el+3);
			stress[k].pt[0].yz += *(stress_el+4);

/* Calculate the principal stresses */

			xxaddyy = .5*(moment[k].pt[0].xx + moment[k].pt[0].yy);
			xxsubyy = .5*(moment[k].pt[0].xx - moment[k].pt[0].yy);
			xysq = moment[k].pt[0].xy*moment[k].pt[0].xy;

			moment[k].pt[0].I = xxaddyy + sqrt( xxsubyy*xxsubyy
			     + xysq);
			moment[k].pt[0].II = xxaddyy - sqrt( xxsubyy*xxsubyy
			     + xysq);

			for( j = 0; j < npel3; ++j )
			{
			    node = *(connect+npel3*k+j);

/* Add all the curvatures and strains for a particular node from all the elements which
   share that node */
			    curve_node[node].xx += curve[k].pt[0].xx;
			    curve_node[node].yy += curve[k].pt[0].yy;
			    curve_node[node].xy += curve[k].pt[0].xy;
			    strain_node[node].zx += strain[k].pt[0].zx;
			    strain_node[node].yz += strain[k].pt[0].yz;
			    curve_node[node].I += curve[k].pt[0].I;
			    curve_node[node].II += curve[k].pt[0].II;

/* Add all the moments and stresses for a particular node from all the elements which
   share that node */
			    moment_node[node].xx += moment[k].pt[0].xx;
			    moment_node[node].yy += moment[k].pt[0].yy;
			    moment_node[node].xy += moment[k].pt[0].xy;
			    stress_node[node].zx += stress[k].pt[0].zx;
			    stress_node[node].yz += stress[k].pt[0].yz;
			    moment_node[node].I += moment[k].pt[0].I;
			    moment_node[node].II += moment[k].pt[0].II;

			}
/*
			printf("%14.6e ",moment[k].pt[0].xx);
			printf("%14.6e ",moment[k].pt[0].yy);
			printf("%14.6e ",moment[k].pt[0].xy);
			printf( "\n");
*/
			/*printf( "\n");*/

/* Calculate the principal straines */

			memset(invariant,0,nsd*sof);
			xysq = strain[k].pt[0].xy*strain[k].pt[0].xy;
			zxsq = strain[k].pt[0].zx*strain[k].pt[0].zx;
			yzsq = strain[k].pt[0].yz*strain[k].pt[0].yz;
			xxyy = strain[k].pt[0].xx*strain[k].pt[0].yy;

			*(invariant) = - strain[k].pt[0].xx -
			     strain[k].pt[0].yy;
			*(invariant+1) = xxyy - yzsq - zxsq - xysq;
			*(invariant+2) = -
			     2*strain[k].pt[0].yz*strain[k].pt[0].zx*strain[k].pt[0].xy +
			     yzsq*strain[k].pt[0].xx + zxsq*strain[k].pt[0].yy;

			check = cubic(invariant);

			strain[k].pt[0].I = *(invariant);
			strain[k].pt[0].II = *(invariant+1);
			strain[k].pt[0].III = *(invariant+2);

			strain_node[node].I += strain[k].pt[0].I;
			strain_node[node].II += strain[k].pt[0].II;
			strain_node[node].III += strain[k].pt[0].III;

/* Calculate the principal stresses */

			memset(invariant,0,nsd*sof);
			xysq = stress[k].pt[0].xy*stress[k].pt[0].xy;
			zxsq = stress[k].pt[0].zx*stress[k].pt[0].zx;
			yzsq = stress[k].pt[0].yz*stress[k].pt[0].yz;
			xxyy = stress[k].pt[0].xx*stress[k].pt[0].yy;

			*(invariant) = - stress[k].pt[0].xx -
			     stress[k].pt[0].yy;
			*(invariant+1) = xxyy - yzsq - zxsq - xysq;
			*(invariant+2) = -
			     2*stress[k].pt[0].yz*stress[k].pt[0].zx*stress[k].pt[0].xy +
			     yzsq*stress[k].pt[0].xx + zxsq*stress[k].pt[0].yy;

			check = cubic(invariant);

			stress[k].pt[0].I = *(invariant);
			stress[k].pt[0].II = *(invariant+1);
			stress[k].pt[0].III = *(invariant+2);

/* Add all the stresses for a particular node from all the elements which share that node */

			stress_node[node].I += stress[k].pt[0].I;
			stress_node[node].II += stress[k].pt[0].II;
			stress_node[node].III += stress[k].pt[0].III;
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

/* Average all the momentes and curves at the nodes */

	  for( i = 0; i < numnp ; ++i )
	  {

		/*printf("%16.8e %16.8e %4d\n", strain_node[i].xx, *(node_counter + i), i);*/
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


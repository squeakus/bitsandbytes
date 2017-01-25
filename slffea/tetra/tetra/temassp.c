/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a tetrahedral element.  It is for
    modal analysis.

		Updated 11/2/06

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
#include "teconst.h"
#include "testruct.h"

extern int numel, numnp, dof, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Vol0;
extern int consistent_mass_flag, consistent_mass_store;

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int tetraB_mass(double *,double *);

int teMassPassemble(int *connect, double *coord, int *el_matl, double *mass,
	MATL *matl, double *P_global, double *U) 
{
        int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
        double B_mass[MsoB], B2_mass[MsoB];
        double M_temp[neqlsq], M_el[neqlsq];
	double U_el[neqel];
        double coord_el_trans[neqel];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double volume_el;
	double P_el[neqel];

	memset(P_global,0,dof*sof);

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (mass + k*neqlsq), U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}
	    }
	}
	else
	{

/* Assemble P matrix by re-deriving element mass matrices */

            for( k = 0; k < numel; ++k )
            {
                matl_num = *(el_matl+k);
                rho = matl[matl_num].rho;

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
                }


/* Assembly of the Mass matrix.

/* Calculation of volume.

   This is taken from "Fundamentals of the Finite Element Method" by
   Hartley Grandin Jr., page 383-387. 

   I have changed the node numbering to be consistant with the
   numbering of the brick element, specifically the counter-clockwise
   numbering scheme.  This means that node 4 in Grandin is node 1 in
   SLFFEA, node 1 becomes node 2, node 2 is node 3, and node 3 goes
   to node 4.  So X4 -> X1, X1 -> X2, X2 -> X3, X3 -> X4, and similarly
   for Y and Z.  This also means a4 -> a1, a1 -> a2, a2 -> a3, a3 -> a4
   and similarly for b and c.  Everything else is the same.
*/

		X1 = *(coord_el_trans);
		X2 = *(coord_el_trans + 1);
		X3 = *(coord_el_trans + 2);
		X4 = *(coord_el_trans + 3);

		Y1 = *(coord_el_trans + npel*1);
		Y2 = *(coord_el_trans + npel*1 + 1);
		Y3 = *(coord_el_trans + npel*1 + 2);
		Y4 = *(coord_el_trans + npel*1 + 3);

		Z1 = *(coord_el_trans + npel*2);
		Z2 = *(coord_el_trans + npel*2 + 1);
		Z3 = *(coord_el_trans + npel*2 + 2);
		Z4 = *(coord_el_trans + npel*2 + 3);

		a2 = (Y3 - Y1)*(Z4 - Z1) - (Y4 - Y1)*(Z3 - Z1);
		a3 = (Y4 - Y1)*(Z2 - Z1) - (Y2 - Y1)*(Z4 - Z1);
		a4 = (Y2 - Y1)*(Z3 - Z1) - (Y3 - Y1)*(Z2 - Z1);
		a1 = -(a2 + a3 + a4);

		b2 = (X4 - X1)*(Z3 - Z1) - (X3 - X1)*(Z4 - Z1);
		b3 = (X2 - X1)*(Z4 - Z1) - (X4 - X1)*(Z2 - Z1);
		b4 = (X3 - X1)*(Z2 - Z1) - (X2 - X1)*(Z3 - Z1);
		b1 = -(b2 + b3 + b4);

		c2 = (X3 - X1)*(Y4 - Y1) - (X4 - X1)*(Y3 - Y1);
		c3 = (X4 - X1)*(Y2 - Y1) - (X2 - X1)*(Y4 - Y1);
		c4 = (X2 - X1)*(Y3 - Y1) - (X3 - X1)*(Y2 - Y1);
		c1 = -(c2 + c3 + c4);

		fdum = (X2 - X1)*a2 + (Y2 - Y1)*b2 + (Z2 - Z1)*c2;

		if( fdum <= 0.0 )
		{
			printf("the element (%d) is inverted; 6*Vol:%f\n", k, fdum);
			return 0;
		}

/* A factor of 1/6 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		volume_el = pt1667*fdum;
		/*printf("This is the Volume %10.6f for element %4d\n",volume_el,k);*/

/* Zero out the Element mass matrices */

        	memset(M_el,0,neqlsq*sof);
		memset(M_temp,0,neqlsq*sof);

/*
   This is taken from "Theory of Matrix Structural Analysis" by 
   J. S. Przemieniecki, page 300-301. 
*/

	fdum = rho*volume_el/20.0;

	*(M_el)    = 2.0*fdum; *(M_el+3)  = 1.0*fdum; *(M_el+6)  = 1.0*fdum; *(M_el+9)  = 1.0*fdum;
	*(M_el+13) = 2.0*fdum; *(M_el+16) = 1.0*fdum; *(M_el+19) = 1.0*fdum; *(M_el+22) = 1.0*fdum;
	*(M_el+26) = 2.0*fdum; *(M_el+29) = 1.0*fdum; *(M_el+32) = 1.0*fdum; *(M_el+35) = 1.0*fdum;
	*(M_el+36) = 1.0*fdum; *(M_el+39) = 2.0*fdum; *(M_el+42) = 1.0*fdum; *(M_el+45) = 1.0*fdum;
	*(M_el+49) = 1.0*fdum; *(M_el+52) = 2.0*fdum; *(M_el+55) = 1.0*fdum; *(M_el+58) = 1.0*fdum;
	*(M_el+62) = 1.0*fdum; *(M_el+65) = 2.0*fdum; *(M_el+68) = 1.0*fdum; *(M_el+71) = 1.0*fdum;
	*(M_el+72) = 1.0*fdum; *(M_el+75) = 1.0*fdum; *(M_el+78) = 2.0*fdum; *(M_el+81) = 1.0*fdum;
	*(M_el+85) = 1.0*fdum; *(M_el+88) = 1.0*fdum; *(M_el+91) = 2.0*fdum; *(M_el+94) = 1.0*fdum;
	*(M_el+98) = 1.0*fdum; *(M_el+101) = 1.0*fdum; *(M_el+104) = 2.0*fdum; *(M_el+107) = 1.0*fdum;
	*(M_el+108) = 1.0*fdum; *(M_el+111) = 1.0*fdum; *(M_el+114) = 1.0*fdum; *(M_el+117) = 2.0*fdum;
	*(M_el+121) = 1.0*fdum; *(M_el+124) = 1.0*fdum; *(M_el+127) = 1.0*fdum; *(M_el+130) = 2.0*fdum;
	*(M_el+134) = 1.0*fdum; *(M_el+137) = 1.0*fdum; *(M_el+140) = 1.0*fdum; *(M_el+143) = 2.0*fdum;


/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, M_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}

	    }
	}

        return 1;
}

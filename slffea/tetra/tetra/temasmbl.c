/*
    This utility function assembles either the lumped sum global
    diagonal Mass matrix or, in the case of consistent mass, all
    the element mass matrices.  This is for a finite element program
    which does analysis on a tetrahedral element.  It is for
    modal analysis.

		Updated 8/22/06

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

#define  DEBUG         0

extern int dof, numel, numnp, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Vol0;
extern int consistent_mass_flag, consistent_mass_store, lumped_mass_flag;

int matXT(double *, double *, double *, int, int, int);

int tetraB_mass(double *,double *);

int teMassemble(int *connect, double *coord, int *el_matl, int *id,
	double *mass, MATL *matl) 
	
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
	double B_mass[MsoB], B2_mass[MsoB];
	double M_temp[neqlsq], M_el[neqlsq];
	double coord_el_trans[neqel];
	double X1, X2, X3, X4, Y1, Y2, Y3, Y4, Z1, Z2, Z3, Z4;
	double a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4;
	double det[1], volume_el, wXdet;
	double mass_el[neqel];

/*      initialize all variables  */

	memcpy(shg,shl,sosh*sizeof(double));

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		rho = matl[matl_num].rho;
		volume_el = 0.0;

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

		*(det) = fdum;

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
		memset(mass_el,0,neqel*sof);

#if !DEBUG
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
#endif

#if 0
		for( i1 = 0; i1 < num_int; ++i1 )
		{
		    for( i2 = 0; i2 < npel; ++i2 )
		    {
			printf("%10.6f ",*(shl+npel*(nsd+1)*i1 + npel*(nsd) + i2));
		    }
		    printf(" \n");
		}
		printf(" \n");
#endif

#if DEBUG

/* The code below is for debugging.  It uses shape functions to calculate
   the B_mass matrix.  

   The loop over j below calculates the 4 points of numerical integration
   for several quantities
*/

		for( j = 0; j < num_int; ++j )
		{

		    memset(B_mass,0,MsoB*sof);
		    memset(B2_mass,0,MsoB*sof);
		    memset(M_temp,0,neqlsq*sof);

/* Assembly of the B matrix for mass */

		    check = tetraB_mass((shg+npel*(nsd+1)*j + npel*(nsd)),B_mass);
		    if(!check) printf( "Problems with tetraB_mass \n");

/*
		    for( i1 = 0; i1 < nsd; ++i1 )
		    {
			for( i2 = 0; i2 < neqel; ++i2 )
			{
				printf("%9.6f ",*(B_mass+neqel*i1+i2));
			}
			printf(" \n");
		    }
		    printf(" \n");
*/

		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

/* A factor of 1/6 is needed to do the integration.  See Eq. 3.I.34 in
   "The Finite Element Method" by Thomas Hughes, page 174
*/
		    wXdet = pt1667*(*(w+j))*(*(det));

		    check=matXT(M_temp, B_mass, B2_mass, neqel, neqel, nsd);
		    if(!check) printf( "Problems with matXT \n");

		    fdum =  rho*wXdet;
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(M_el+i2) += *(M_temp+i2)*fdum;
		    }
		}
#endif

		if(lumped_mass_flag)
		{

/* Creating the diagonal lumped mass Matrix */

		    fdum = 0.0;
		    for( i2 = 0; i2 < neqel; ++i2 )
		    {   
			/*printf("This is mass_el for el %3d",k);*/
			for( i3 = 0; i3 < neqel; ++i3 )
			{
			    *(mass_el+i2) += *(M_el+neqel*i2+i3);
			}
			/*printf("%9.6f\n\n",*(mass_el+i2));*/
			fdum += *(mass_el+i2);
		    }   
		    /* printf("This is Volume2 %9.6f\n\n",fdum);*/

		    for( j = 0; j < neqel; ++j )
		    {   
			*(mass+*(dof_el+j)) += *(mass_el + j);
		    }
		}

		if(consistent_mass_flag)
		{

/* Storing all the element mass matrices */

		    for( j = 0; j < neqlsq; ++j )
		    {   
			*(mass + neqlsq*k + j) = *(M_el + j);
		    }
		}
	}

	if(lumped_mass_flag)
	{
/* Contract the global mass matrix using the id array only if lumped
   mass is used. */

	    counter = 0;
	    for( i = 0; i < dof ; ++i )
	    {
		/* printf("%5d  %16.8e\n", i, *(mass+i));*/
		if( *(id + i ) > -1 )
		{
		    *(mass + counter ) = *(mass + i );
		    ++counter;
		}
	    }
	}

	return 1;
}

/*
    This utility function assembles the Mass and force matrix for a finite
    element program which does analysis on a truss which can behave
    nonlinearly.  The mass is used both in the dynamic relaxation
    method as well as the conjugate gradient method.   It is not really
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

    Note that I do not use Lh for K_local in ts2Passemble when the conjugate
    gradient method is used.  This is because I want to keep things consistent
    with weConjPassemble.


        Updated 3/15/06

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
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"

extern int numel, numnp, dof, sof;

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int tsFMassemble(int *connect, double *coord, double *coordh, int *el_matl,
        double *force, double *mass, MATL *matl, double *U)
{
        int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node0, node1;
        int matl_num;
        double Emod, area, EmodXarea;
        double L, Lx, Ly, Lz, Lsq, Lh, Lxh, Lyh, Lzh, Lhsq;
        double jacob, jacobh;
	double K_temp[npel*neqel], K_el[neqlsq], K_local[npelsq], rotate[npel*neqel];
	double force_el[neqel], force_ax[npel], U_el[neqel], U_ax[npel];
        double stress_el[sdim], strain_el[sdim];
	double x[num_int], w[num_int];
	double mass_el[neqel];

/* initialize all variables  */
	memset(mass,0,dof*sof);

        *(x)=0.0;
        *(w)=2.0;

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		area = matl[matl_num].area;
		EmodXarea = Emod*area;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

/* Calculate half time quantites for one element */

		Lxh = *(coordh+nsd*node1) - *(coordh+nsd*node0);
		Lyh = *(coordh+nsd*node1+1) - *(coordh+nsd*node0+1);
		Lzh = *(coordh+nsd*node1+2) - *(coordh+nsd*node0+2);

		Lhsq = Lxh*Lxh + Lyh*Lyh + Lzh*Lzh;
		Lh = sqrt(Lhsq);
		Lxh /= Lh; Lyh /= Lh; Lzh /= Lh;

		jacobh = Lh/2.0;

/* Calculate full time quantites for one element */

		Lx = *(coord+nsd*node1) - *(coord+nsd*node0);
		Ly = *(coord+nsd*node1+1) - *(coord+nsd*node0+1);
		Lz = *(coord+nsd*node1+2) - *(coord+nsd*node0+2);

		Lsq = Lx*Lx+Ly*Ly+Lz*Lz;
		L = sqrt(Lsq);
		Lx /= L; Ly /= L; Lz /= L;

		jacob = L/2.0;

/* Assembly of the rotation matrix.  In the linear truss, I 
   pre-assemble the rotation matrix with the length to expediate 
   calculations.  This cannot be done for the non-linear truss  
   though, because the coordinate are being continually updated  
   with each time step.
*/

		memset(rotate,0,npel*neqel*sof);
		*(rotate) = Lxh;
		*(rotate+1) = Lyh;
		*(rotate+2) = Lzh;
		*(rotate+9) = Lxh;
		*(rotate+10) = Lyh;
		*(rotate+11) = Lzh;

/* defining the components of an element vector */

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(mass_el,0,neqel*sof);
		memset(force_el,0,neqel*sof);
		memset(force_ax,0,npel*sof);
		memset(K_temp,0,npel*neqel*sof);
		memset(K_local,0,npel*npel*sof);

/* Assembly of the local stiffness matrix */

		*(K_local) = EmodXarea/L/Lh;
		*(K_local+1) = - EmodXarea/L/Lh;
		*(K_local+2) = - EmodXarea/L/Lh;
		*(K_local+3) = EmodXarea/L/Lh;

		for( i1 = 0; i1 < npelsq; ++i1 )
		{
		    *(K_el + i1) += *(K_local + i1)*jacob*(*(w));
		}

/* Put K back to global coordinates */

		check = matX(K_temp, K_el, rotate, npel, neqel, npel);
		if(!check) printf( "Problems with matX \n");

		check = matXT(K_el, rotate, K_temp, neqel, neqel, npel);
		if(!check) printf( "Problems with matXT \n");

		*(U_el) = *(U + *(dof_el));
		*(U_el+1) = *(U + *(dof_el+1));
		*(U_el+2) = *(U + *(dof_el+2));
		*(U_el+3) = *(U + *(dof_el+3));
		*(U_el+4) = *(U + *(dof_el+4));
		*(U_el+5) = *(U + *(dof_el+5));

		check = matX(force_el, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

/* Compute the equivalant nodal forces based on prescribed displacements */

		for( j = 0; j < neqel; ++j )
		{
			*(force + *(dof_el+j)) -= *(force_el + j);
		}

/* Creating the mass Matrix */

		/*for( i3 = 0; i3 < neqel; ++i3 )
		{
		  printf( "\n %4d ",i3);
		  for( i2 = 0; i2 < neqel; ++i2 )
		  {
		   printf( "%16.4e ",*(K_el+neqel*i3+i2));
		  }
		}*/

		printf( "\n");
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
	for( i = 0; i < dof ; ++i )
	{
		printf( " force %4d %16.8e \n",i,*(force+i));
	}

	return 1;
}


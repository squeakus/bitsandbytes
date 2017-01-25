/*
    This utility function assembles the P_global matrix for a finite
    element program which does analysis on a truss which can behave
    nonlinearly.  It can also do assembly for the P_global_CG matrix
    if non-linear analysis using the conjugate gradient method is chosen.

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
extern int Passemble_flag, Passemble_CG_flag;

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int ts2Passemble(int *connect, double *coord, double *coordh, int *el_matl,
	MATL *matl, double *P_global, SDIM *stress, double *dU,
	double *P_global_CG, double *U)
{
	int i, i1, i2, j, k, dof_el[neqel];
	int check, node0, node1;
	int matl_num;
	double Emod, area, EmodXarea, wXjacobXarea;
	double L, Lx, Ly, Lz, Lsq, Lh, Lxh, Lyh, Lzh, Lhsq;
	double B[soB], Bh[soB], DB[soB], jacob, jacobh;
	double P_el[neqel], P_temp[neqel], P_local[npel];
	double K_temp[npel*neqel], K_el[neqlsq], K_local[npelsq], rotate[npel*neqel];
	double dU_el[neqel], U_el[neqel], dU_ax[npel];
	double stress_el[sdim], dstress_el[sdim], dstrain_el[sdim];
	double x[num_int], w[num_int];
	int i4,i5,i5m1,i5m2;

	*(x)=0.0;
	*(w)=2.0;

	memset(P_global,0,dof*sof);
	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		area = matl[matl_num].area;
		EmodXarea = Emod*area;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

/* defining the components of an element vector */

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

		*(dU_el) = *(dU + ndof*node0);
		*(dU_el+1) = *(dU + ndof*node0+1);
		*(dU_el+2) = *(dU + ndof*node0+2);
		*(dU_el+3) = *(dU + ndof*node1);
		*(dU_el+4) = *(dU + ndof*node1+1);
		*(dU_el+5) = *(dU + ndof*node1+2);

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

		Lsq = Lx*Lx + Ly*Ly + Lz*Lz;
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
		*(rotate) = Lx;
		*(rotate+1) = Ly;
		*(rotate+2) = Lz;
		*(rotate+9) = Lx;
		*(rotate+10) = Ly;
		*(rotate+11) = Lz;

		memset(P_el,0,neqel*sof);
		if(Passemble_CG_flag)
		{
		   memset(U_el,0,neqel*sof);
		   memset(K_el,0,neqlsq*sof);
		   memset(K_temp,0,npel*neqel*sof);
		   memset(K_local,0,npel*npel*sof);
		}

		if(Passemble_flag)
		{
		   memset(B,0,soB*sof);
		   memset(Bh,0,soB*sof);

/* Assembly of the Bh matrix at 1/2 time */

		   *(Bh) = - 1.0/Lh;
		   *(Bh+1) = 1.0/Lh;

/* Assembly of the B matrix at full time */

		   *(B) = - 1.0/L;
		   *(B+1) = 1.0/L;

#if 0

/* Because trusses are one dimensional elements in the sense that there is
   only axial deformation, stress can be defined using the change in the length
   of an element.  Because of the improved accuracy and speed, I do this for both
   the dynamic relaxation method and the Conjugate Gradient method.  The lines
   commented out below represent the traditional way of calculating incremental
   strain and stress and then updated the global stress.
*/
   
/* Calculation of the incremental strain at 1/2 time */

		   check = matX(dU_ax, rotate, dU_el, npel, 1, neqel);
		   if(!check) printf( "Problems with matX \n");

		   *(dstrain_el) = *(Bh)*(*(dU_ax)) + *(Bh+1)*(*(dU_ax+1));

/* Calculation of the incremental stress rate change */

		   *(dstress_el) = *(dstrain_el)*Emod;

/* Update the global stress matrix */

		   stress[k].xx += *(dstress_el);
#endif

/* Calculation of the element stress matrix at full time */

		   *(stress_el) = stress[k].xx;

		}

		if(Passemble_flag)
		{
/* Assembly of the local element P matrix = (B transpose)*stress_el  */

		   *(P_local) = *(B)*(*(stress_el));
		   *(P_local+1) = *(B+1)*(*(stress_el));

/* Put P back to global coordinates */

		   check = matXT(P_temp, rotate, P_local, neqel, 1, npel);
		   if(!check) printf( "Problems with matXT \n");

		   wXjacobXarea = *(w)*jacobh*area;
		   for( i1 = 0; i1 < neqel; ++i1 )
		   {
			/*printf("rrrrr %14.9f \n",*(P_temp+i1));*/
			/*printf("rrrrr %4d %4d %14.9f \n", k, i1, *(P_temp+i1)*wXjacobXarea);*/
			*(P_el+i1) += *(P_temp+i1)*wXjacobXarea;
		   }
		}

		if(Passemble_CG_flag)
		{
/* Assembly of the local stiffness matrix */

/*
    Note that I do not use Lh for K_local when the conjugate gradient
    method is used.  This is because I want to keep things consistent with
    tsConjPassemble.
*/


		    *(K_local) = EmodXarea/L/L;
		    *(K_local+1) = - EmodXarea/L/L;
		    *(K_local+2) = - EmodXarea/L/L;
		    *(K_local+3) = EmodXarea/L/L;

		    for( i1 = 0; i1 < npelsq; ++i1 )
		    {
			*(K_el + i1) += *(K_local + i1)*jacob*(*(w));
		    }

/* Put K back to global coordinates */

		    check = matX(K_temp, K_el, rotate, npel, neqel, npel);
		    if(!check) printf( "Problems with matX \n");

		    check = matXT(K_el, rotate, K_temp, neqel, neqel, npel);
		    if(!check) printf( "Problems with matXT \n");
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

		    memset(P_el,0,neqel*sof);
		    check = matX(P_el, K_el, U_el, neqel, 1, neqel);
		    if(!check) printf( "Problems with matX \n");

		    for( j = 0; j < neqel; ++j )
		    {
			*(P_global_CG + *(dof_el+j)) += *(P_el+j);
		    }
		}
	}

	return 1;
}


/*
    This library function assembles the stiffness matrix and calculates the
    reaction forces for a finite element program which does analysis on a truss.

                     Updated 11/8/06

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
#include "tsconst.h"
#include "tsstruct.h"

extern int analysis_flag, dof, neqn, numel, numnp, sof;
extern int LU_decomp_flag, numel_K, numel_P;

int globalConjKassemble(double *, int *, int , double *,
	double *, int , int , int );

int globalKassemble(double *, int *, double *, int *, int );

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int tsKassemble(double *A, int *connect, double *coord, int *el_matl, double *force,
	int *id, int *idiag, double *K_diag, int *lm, double *length, double *local_xyz,
	MATL *matl, SDIM *strain, SDIM *stress, double *U)
{
	int i, i1, i2, j, ij, k, dof_el[neqel];
	int check, counter, node0, node1;
	int matl_num;
	double Emod, area, EmodXarea, wXjacob;
	double L, Lx, Ly, Lz, Lsq;
	double B[soB], DB[soB], jacob;
	double K_temp[npel*neqel], K_el[neqlsq], K_local[npelsq], rotate[npel*neqel];
	double force_el[neqel], force_ax[npel], U_el[neqel], U_ax[npel];
	double stress_el[sdim], strain_el[sdim];
	double x[num_int], w[num_int];

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

/* defining the components of an element vector */

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

		L = *(length + k);
		Lsq = L*L;

		jacob = L/2.0;

/* Assembly of the rotation matrix.  This is taken from equation 5.42 on page 69 of:

     Przemieniecki, J. S., Theory of Matrix Structural Analysis, Dover
        Publications Inc., New York, 1985.
*/

		memset(rotate,0,npel*neqel*sof);

		*(rotate)    = *(local_xyz + nsdsq*k);
		*(rotate+1)  = *(local_xyz + nsdsq*k + 1);
		*(rotate+2)  = *(local_xyz + nsdsq*k + 2);
		*(rotate+9)  = *(local_xyz + nsdsq*k);
		*(rotate+10) = *(local_xyz + nsdsq*k + 1);
		*(rotate+11) = *(local_xyz + nsdsq*k + 2);

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);
		memset(force_el,0,neqel*sof);
		memset(force_ax,0,npel*sof);
		memset(K_temp,0,npel*neqel*sof);
		memset(B,0,soB*sof);
		memset(DB,0,soB*sof);
		memset(K_local,0,npel*npel*sof);

/* Assembly of the local stiffness matrix */

		*(B) = - 1.0/L;
		*(B+1) = 1.0/L;

		*(DB) = Emod*(*(B));
		*(DB+1) = Emod*(*(B+1));

		*(K_local) = EmodXarea/L/L;
		*(K_local+1) = - EmodXarea/L/L;
		*(K_local+2) = - EmodXarea/L/L;
		*(K_local+3) = EmodXarea/L/L;

		wXjacob = *(w)*jacob;
		for( i1 = 0; i1 < npelsq; ++i1 )
		{
		    *(K_el + i1) += *(K_local + i1)*wXjacob;
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

		if(analysis_flag == 1)
		{

/* Compute the equivalant nodal forces based on prescribed displacements */

			for( j = 0; j < neqel; ++j )
			{
				*(force + *(dof_el+j)) -= *(force_el + j);
			}

/* Assembly of either the global skylined stiffness matrix or numel_K of the
   element stiffness matrices if the Conjugate Gradient method is used */

			if(LU_decomp_flag)
			{
			    check = globalKassemble(A, idiag, K_el, (lm + k*neqel),
				neqel);
			    if(!check) printf( "Problems with globalKassemble \n");
			}
			else
			{
			    check = globalConjKassemble(A, dof_el, k, K_diag, K_el,
				neqel, neqlsq, numel_K);
			    if(!check) printf( "Problems with globalConjKassemble \n");
			}
		}
		else
		{
/* Calculate the element reaction forces */

			for( j = 0; j < neqel; ++j )
			{
				*(force + *(dof_el+j)) += *(force_el + j);
			}

/* Calculate the element axial forces */

#if DATA_ON
			printf("\n element (%3d)  node %3d       node %3d",
				k,node0,node1);
#endif
			check = matX(U_ax, rotate, U_el, npel, 1, neqel);
			if(!check) printf( "Problems with matX \n");
#if DATA_ON
			printf("\n displacement  %9.5f      %9.5f",
				*(U_ax), *(U_ax+1));
#endif
			check = matX(force_ax, K_local, U_ax, npel, 1, npel);
			if(!check) printf( "Problems with matX \n");
			*(force_ax) *= L;
			*(force_ax+1) *= L;
#if DATA_ON
			printf("\n force    %14.5f %14.5f",
				*(force_ax), *(force_ax+1));
#endif

			memset(stress_el,0,sdim*sof);
			memset(strain_el,0,sdim*sof);

/* Calculation of the local strain matrix */

			check=matX(strain_el, B, U_ax, sdim, 1, npel );
			if(!check) printf( "Problems with matX \n");

/* Update of the global strain matrix */

			strain[k].xx = *(strain_el);

/* Calculation of the local stress matrix */

			*(stress_el) = strain[k].xx*Emod;

/* Update of the global stress matrix */

			stress[k].xx += *(stress_el);

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

	return 1;
}


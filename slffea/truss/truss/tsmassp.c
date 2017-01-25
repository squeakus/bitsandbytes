/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a truss.  It is for modal analysis.

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

extern int numel, numnp, dof, sof;
extern int consistent_mass_flag, consistent_mass_store;

int mat(double *, double *, double *, int, int, int);

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int tsMassPassemble(int *connect, double *coord, int *el_matl, double *length,
	double *local_xyz, double *mass, MATL *matl, double *P_global,
	double *U) 
{
	int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, counter, node0, node1;
	int matl_num;
	double area, rho, fdum, fdum2;
	double L, Lsq;
	double M_temp[neqlsq], M_el[neqlsq], rotate[nsdsq];
	double U_el[neqel];
	double jacob;
	double P_el[neqel];

	memset(P_global,0,dof*sof);

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

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
		area = matl[matl_num].area;

		node0 = *(connect+k*npel);
		node1 = *(connect+k*npel+1);

		L = *(length + k);
		Lsq = L*L;
		jacob = L/2.0;

/* Assembly of the 3X3 rotation matrix for the 6X6 global rotation
   matrix */

		memset(rotate,0,nsdsq*sof);

		*(rotate)     = *(local_xyz + nsdsq*k);
		*(rotate + 1) = *(local_xyz + nsdsq*k + 1);
		*(rotate + 2) = *(local_xyz + nsdsq*k + 2);
		*(rotate + 3) = *(local_xyz + nsdsq*k + 3);
		*(rotate + 4) = *(local_xyz + nsdsq*k + 4);
		*(rotate + 5) = *(local_xyz + nsdsq*k + 5);
		*(rotate + 6) = *(local_xyz + nsdsq*k + 6);
		*(rotate + 7) = *(local_xyz + nsdsq*k + 7);
		*(rotate + 8) = *(local_xyz + nsdsq*k + 8);

/* defining the components of a el element vector */

		*(dof_el)=ndof*node0;
		*(dof_el+1)=ndof*node0+1;
		*(dof_el+2)=ndof*node0+2;
		*(dof_el+3)=ndof*node1;
		*(dof_el+4)=ndof*node1+1;
		*(dof_el+5)=ndof*node1+2;

		fdum = .5*rho*area*L;

		memset(M_el,0,neqlsq*sof);
		memset(M_temp,0,neqlsq*sof);

/* The mass matrix below comes from equation 10.81 on page 279 of:

     Przemieniecki, J. S., Theory of Matrix Structural Analysis, Dover
        Publications Inc., New York, 1985.
*/

/* row 0 */
		*(M_el) = 1.0/3.0;
		*(M_el+3) = 1.0/6.0;
		*(M_el+18) = *(M_el+3);
/* row 1 */
		*(M_el+7) = 1.0/3.0;
		*(M_el+10) = 1.0/6.0;
		*(M_el+25) = *(M_el+10);
/* row 2 */
		*(M_el+14) = 1.0/3.0;
		*(M_el+17) = 1.0/6.0;
		*(M_el+32) = *(M_el+17);
/* row 3 */
		*(M_el+21) = 1.0/3.0;
/* row 4 */
		*(M_el+28) = 1.0/3.0;
/* row 5 */
		*(M_el+35) = 1.0/3.0;

		fdum = rho*area*L;
		for( i1 = 0; i1 < neqlsq; ++i1 )
		{
			*(M_el + i1) = *(M_el + i1)*fdum;
		}

/* Put M_el back to global coordinates */

		check = matXrot(M_temp, M_el, rotate, neqel, neqel);
		if(!check) printf( "error with matXrot \n");

		check = rotTXmat(M_el, rotate, M_temp, neqel, neqel);
		if(!check) printf( "error with rotTXmat \n");

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

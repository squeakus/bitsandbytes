/*
    This utility function assembles the lumped sum Mass matrix for a
    finite element program which does analysis on a truss.  It is for
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
#include "tsconst.h"
#include "tsstruct.h"

extern int dof, numel, numnp, sof;
extern int consistent_mass_flag, consistent_mass_store, lumped_mass_flag;

int matX(double *, double *, double *, int, int, int);

int matXT(double *, double *, double *, int, int, int);

int matXrot(double *, double *, double *, int, int);

int rotXmat(double *, double *, double *, int, int);

int tsMassemble( int *connect, double *coord, int *el_matl, int *id, double *length,
        double *local_xyz, double *mass, MATL *matl)
{
	int i, i1, i2, i3, j, k, ij, dof_el[neqel];
	int check, counter, dum, node0, node1;
	int matl_num, el_num, type_num;
	double area, rho, fdum, fdum2;
	double L, Lsq;
	double M_temp[neqlsq], M_el[neqlsq], mass_el[neqel],
		mass_local[neqlsq], rotate[nsdsq];
	double jacob;

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
		memset(mass_el,0,neqel*sof);

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
		    /*printf("This is Volume2 %9.6f\n\n",fdum);*/

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


/*
    This utility function takes the product of a vector with the
    either the Btt, Btz, or Bzt matrix.  This is for a finite element program
    which does analysis on a quad.  It is for modal analysis.

		Updated 11/11/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "qd3const.h"
#include "qd3struct.h"

extern int numed, numel, numnp, dof, EMdof, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Area0;
extern int B_matrix_store;

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int quadB_mass(double *,double *);

int qdshg_mass( double *, int, double *, double *, double *);

int qdBPassemble(int *connect, double *coord, int *el_matl, double *B_matrix,
	MATL *matl, double *P_global, double *U) 
{
        int i, i1, i2, i3, j, k, dof_el[neqel], EMdof_el[EMneqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
        double B_mass[EMsoB], B2_mass[EMsoB];
        double M_temp[EMneqlsq], M_el[EMneqlsq];
	double U_el[EMneqel];
        double coord_el_trans[EMneqel];
        double det[num_int];
	double P_el[EMneqel];

	memset(P_global,0,EMdof*sof);
	memset(B_mass,0,EMsoB*sof);
	memset(B2_mass,0,EMsoB*sof);

	memcpy(shg,shl,sosh*sizeof(double));

	if(B_matrix_store)
	{

/* Assemble P matrix using stored element B_matrix matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < EMneqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (B_matrix + k*EMneqlsq), U_el, EMneqel, 1, EMneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < EMneqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}
	    }
	}
	else
	{

/* Assemble P matrix by re-deriving element B_matrix matrices */

            for( k = 0; k < numel; ++k )
            {
                matl_num = *(el_matl+k);
                rho = matl[matl_num].eta;

/* Zero out the Element and B_matrix matrices */

        	memset(M_el,0,EMneqlsq*sof);

/* Create the coord_el transpose vector for one element */

                for( j = 0; j < npel; ++j )
                {
			node = *(connect+npel*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel*1+j)=*(coord+*(sdof_el+nsd*j+1));

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
                }

/* The call to qdshg_mass is only for calculating the determinent */

		check = qdshg_mass(det, k, shg, coord_el_trans, &fdum);
       		if(!check) printf( "Problems with qdshg_mass \n");

#if 0
                for( i1 = 0; i1 < num_int; ++i1 )
                {
                    for( i2 = 0; i2 < npel; ++i2 )
                    {
                    	printf("%10.6f ",*(shl+npel*(nsd+1)*i1 + npel*(nsd) + i2));
                    }
                    printf(" \n ");
                }
                printf(" \n ");
#endif

/* The loop over j below calculates the 8 points of the gaussian integration
   for several quantities */

                for( j = 0; j < num_int; ++j )
                {

/* Assembly of the B matrix for B_matrix */

       		    check = quadB_mass((shg+npel*(nsd+1)*j + npel*(nsd)),B_mass);
       		    if(!check) printf( "Problems with quadB_mass \n");

#if 0
                    for( i1 = 0; i1 < nsd; ++i1 )
                    {
                        for( i2 = 0; i2 < EMneqel; ++i2 )
                        {
                        	printf("%9.6f ",*(B_mass+EMneqel*i1+i2));
                        }
                        printf(" \n ");
                    }
                    printf(" \n ");
#endif
		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

                    check=matXT(M_temp, B_mass, B2_mass, EMneqel, EMneqel, nsd);
       		    if(!check) printf( "Problems with matXT \n");

		    fdum = rho*(*(w+j))*(*(det+j));
                    for( i2 = 0; i2 < EMneqlsq; ++i2 )
                    {
                          *(M_el+i2) += *(M_temp+i2)*fdum;
                    }
                }

/* Assembly of the global P matrix */

		for( j = 0; j < EMneqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, M_el, U_el, EMneqel, 1, EMneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < EMneqel; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}

	    }
	}

        return 1;
}

/*
    This utility function assembles either the lumped sum global
    diagonal Mass matrix or, in the case of consistent mass, all
    the element mass matrices.  This is for a finite element program
    which does analysis on a quad.  It is for modal analysis.

		Updated 10/9/00

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
#include "qdconst.h"
#include "qdstruct.h"

extern int dof, numel, numnp, sof;
extern double shg[sosh], shl[sosh], w[num_int], *Area0;
extern int consistent_mass_flag, B_matrix_store, lumped_mass_flag;

int matXT(double *, double *, double *, int, int, int);

int quadB_mass(double *,double *);

int qdshg_mass( double *, int, double *, double *, double *);

int qdMassemble(int *connect, double *coord, int *el_matl, int *id,
	double *mass, MATL *matl) 
	
{
        int i, i1, i2, i3, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum;
        double B_mass[MsoB], B2_mass[MsoB];
        double M_temp[neqlsq], M_el[neqlsq];
        double coord_el_trans[neqel];
        double det[num_int], volume_el;
        double mass_el[neqel];

/*      initialize all variables  */

        memset(B_mass,0,MsoB*sof);
        memset(B2_mass,0,MsoB*sof);

	memcpy(shg,shl,sosh*sizeof(double));

        for( k = 0; k < numel; ++k )
        {
                matl_num = *(el_matl+k);
                rho = matl[matl_num].rho;

/* Zero out the Element and mass matrices */

        	memset(M_el,0,neqlsq*sof);
        	memset(mass_el,0,neqel*sof);

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

		check = qdshg_mass(det, k, shg, coord_el_trans, &volume_el);
		if(!check) printf( "Problems with qdshg_mass \n");

                /* printf("This is 2 X Area %10.6f for element %4d\n",2.0*volume_el,k);*/

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

/* The loop over j below calculates the 8 points of the gaussian integration
   for several quantities */

                for( j = 0; j < num_int; ++j )
                {

/* Assembly of the B matrix for mass */

       		    check = quadB_mass((shg+npel*(nsd+1)*j + npel*(nsd)),B_mass);
       		    if(!check) printf( "Problems with quadB_mass \n");

#if 0
                    for( i1 = 0; i1 < nsd; ++i1 )
                    {
                        for( i2 = 0; i2 < neqel; ++i2 )
                        {
                        	printf("%9.6f ",*(B_mass+neqel*i1+i2));
                        }
                        printf(" \n");
                    }
                    printf(" \n");
#endif

		    memcpy(B2_mass,B_mass,MsoB*sizeof(double));

                    check=matXT(M_temp, B_mass, B2_mass, neqel, neqel, nsd);
                    if(!check) printf( "Problems with matXT \n");

		    fdum = rho*(*(w+j))*(*(det+j));
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			*(M_el+i2) += *(M_temp+i2)*fdum;
		    }
		}

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
		    /*printf("This is Area2 %9.6f\n\n",fdum);*/

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

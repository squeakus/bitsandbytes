/*
    This utility function takes the product of a vector with the
    consistent mass matrix.  This is for a finite element program
    which does analysis on a shell.  It is for modal analysis.

		Updated 10/10/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "shconst.h"
#include "shstruct.h"

extern int numel, numnp, dof, sof, flag_quad_element;
extern SH shg, shl, shl_tri;
extern ROTATE rotate, rotate_node;
extern double w[num_int8], w_tri[num_int6], *Vol0;
extern int consistent_mass_flag, consistent_mass_store;

int matXT(double *, double *, double *, int, int, int);

int matX(double *, double *, double *, int, int, int);

int shellB_mass(double *, double *, double *, double *);

int shshg_mass( double *, int , SH , XL , double *, double *, double *,
	double *);

int shMassPassemble(int *connect, double *coord, int *el_matl, double *lamina_ref,
	double *fiber_xyz, double *mass, MATL *matl, double *P_global, double *U) 
{
	int i, i1, i2, i3, i4, j, k, dof_el[neqel20], sdof_el[npel8*nsd];
	int check, node, counter;
	int matl_num;
	double rho, fdum, fdum1, fdum2, fdum3, fdum4;
	double B_mass[MsoB60], B2_mass[MsoB60];
	double M_temp[neqlsq400], M_el[neqlsq400];
	double U_el[neqel20];
	double coord_el_trans[npel8*nsd], zm1[npell4], zp1[npell4],
		znode[npell4*num_ints], dzdt_node[npell4];
	double det[num_int8];
	XL xl;
	double P_el[neqel20];
        int npell_, neqel_, num_intb_, neqlsq_, soB_, npel_, MsoB_;
        SH *shl_;

	npell_ = npell4;
	neqel_ = neqel20;
	num_intb_ = num_intb4;
	neqlsq_ = neqlsq400;
	soB_ = soB100;
	npel_ = npel8;
	MsoB_ = MsoB60;
	shl_ = &shl;

	if(!flag_quad_element)
	{
		npell_ = npell3;
		neqel_ = neqel15;
		num_intb_ = num_intb3;
		neqlsq_ = neqlsq225;
		soB_ = soB75;
		npel_ = npel6;
		MsoB_ = MsoB45;
		shl_ = &shl_tri;
	}

	memset(P_global,0,dof*sof);

	if(consistent_mass_store)
	{

/* Assemble P matrix using stored element mass matrices */

	    for( k = 0; k < numel; ++k )
	    {

		for( j = 0; j < npell_; ++j )
		{
			node = *(connect+npell_*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
			*(dof_el+ndof*j+3) = ndof*node+3;
			*(dof_el+ndof*j+4) = ndof*node+4;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel_; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (mass + k*neqlsq_), U_el, neqel_, 1, neqel_);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel_; ++j )
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

/* Create the coord_el transpose vector and other variables for one element */

		for( j = 0; j < npell_; ++j )
		{
			node = *(connect+npell_*k+j);

			*(sdof_el+nsd*j)=nsd*node;
			*(sdof_el+nsd*j+1)=nsd*node+1;
			*(sdof_el+nsd*j+2)=nsd*node+2;

			*(sdof_el+nsd*npell_+nsd*j)=nsd*(node+numnp);
			*(sdof_el+nsd*npell_+nsd*j+1)=nsd*(node+numnp)+1;
			*(sdof_el+nsd*npell_+nsd*j+2)=nsd*(node+numnp)+2;

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
			*(dof_el+ndof*j+3) = ndof*node+3;
			*(dof_el+ndof*j+4) = ndof*node+4;

/* Create the coord -/+*/

			*(coord_el_trans+j) =
				*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel_*1+j) =
				*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel_*2+j) =
				*(coord+*(sdof_el+nsd*j+2));

			*(coord_el_trans+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j));
			*(coord_el_trans+npel_*1+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j+1));
			*(coord_el_trans+npel_*2+npell_+j) =
				*(coord+*(sdof_el+nsd*npell_+nsd*j+2));

/* Create the coord_bar and coord_hat vector for one element */

			xl.bar[j] = *(lamina_ref + nsd*node);
			xl.bar[npell_*1+j] = *(lamina_ref + nsd*node + 1);
			xl.bar[npell_*2+j] = *(lamina_ref + nsd*node + 2);

			fdum1=*(coord_el_trans+npell_+j)-*(coord_el_trans+j);
			fdum2=*(coord_el_trans+npel_*1+npell_+j)-*(coord_el_trans+npel_*1+j);
			fdum3=*(coord_el_trans+npel_*2+npell_+j)-*(coord_el_trans+npel_*2+j);
			fdum4=sqrt(fdum1*fdum1+fdum2*fdum2+fdum3*fdum3);

			*(zp1+j)=.5*(1.0-zeta)*fdum4;
			*(zm1+j)=-.5*(1.0+zeta)*fdum4;

			xl.hat[j] = *(fiber_xyz + nsdsq*node + 2*nsd);
			xl.hat[npell_*1+j] = *(fiber_xyz + nsdsq*node + 2*nsd + 1);
			xl.hat[npell_*2+j] = *(fiber_xyz + nsdsq*node + 2*nsd + 2);


/* Create the rotation matrix */

			for( i1 = 0; i1 < nsd; ++i1 )
			{
			    rotate.f_shear[nsdsq*j + i1] =
				*(fiber_xyz + nsdsq*node + i1);
			    rotate.f_shear[nsdsq*j + 1*nsd + i1] =
				*(fiber_xyz + nsdsq*node + 1*nsd + i1);
			    rotate.f_shear[nsdsq*j + 2*nsd + i1] =
				*(fiber_xyz + nsdsq*node + 2*nsd + i1);
			}
		}

/* The call to shshg_mass is only for calculating the determinent */

		check = shshg_mass( det, k, *shl_, xl, zp1, zm1, znode,
			dzdt_node);
		if(!check) printf( "Problems with shshg_mass \n");

#if 0
		for( i1 = 0; i1 < num_intb_; ++i1 )
		{
		    for( i2 = 0; i2 < npell_; ++i2 )
		    {
			printf("%10.6f ",*(shl+npell_*(nsd+1)*i1 + npell_*(nsd) + i2));
		    }
		    printf(" \n");
		}
		printf(" \n");
#endif

/* Loop through the integration points */

/* Zero out the Element mass matrices */

		memset(M_el,0,neqlsq_*sof);

/* The loop over i4 below calculates the 2 fiber points of the gaussian integration */
		for( i4 = 0; i4 < num_ints; ++i4 )
		{

/* The loop over j below calculates the 2X2 points of the gaussian integration
   over the lamina for several quantities */

		   for( j = 0; j < num_intb_; ++j )
		   {
			memset(B_mass,0,MsoB_*sof);
			memset(B2_mass,0,MsoB_*sof);
			memset(M_temp,0,neqlsq_*sof);

/* Assembly of the B matrix for mass */

			check =
			    shellB_mass((shg.bend+npell_*(nsd+1)*num_intb_*i4+npell_*(nsd+1)*j+npell_*(nsd)),
			    znode, B_mass, rotate.f_shear);
			if(!check) printf( "Problems with shellB_mass \n");
#if 0
			for( i1 = 0; i1 < nsd; ++i1 )
			{
			    for( i2 = 0; i2 < neqel_; ++i2 )
			    {
				printf("%9.6f ",*(B_mass+neqel_*i1+i2));
			    }
			    printf(" \n");
			}
			printf(" \n");
#endif

			memcpy(B2_mass,B_mass,MsoB_*sizeof(double));

			check=matXT(M_temp, B_mass, B2_mass, neqel_, neqel_, nsd);
			if(!check) printf( "Problems with matXT \n");

			fdum =  rho*(*(w+num_intb_*i4+j))*(*(det+num_intb_*i4+j));
			if(!flag_quad_element)
			    fdum = rho*0.5*(*(w_tri+num_intb_*i4+j))*(*(det+num_intb_*i4+j));

			for( i2 = 0; i2 < neqlsq_; ++i2 )
			{
				*(M_el+i2) += *(M_temp+i2)*fdum;
			}
		   }
		}

		/*printf("\n %4d \n", k);
		for( i2 = 0; i2 < neqel_; ++i2 )
		{
			for( i4 = neqel_-5; i4 < neqel_; ++i4 )
			{
				printf(" %16.8e", *(M_el+i2));
			}
			printf("\n");
		}*/

/* Assembly of the global P matrix */

		for( j = 0; j < neqel_; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, M_el, U_el, neqel_, 1, neqel_);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel_; ++j )
		{
			*(P_global+*(dof_el+j)) += *(P_el+j);
		}

	    }
	}

	return 1;
}

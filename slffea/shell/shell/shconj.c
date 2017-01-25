/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a shell.  The first function
    assembles the P matrix.  It is called by the second function
    which allocates the memory and goes through the steps of the algorithm.
    These go with the calculation of displacement.

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

#define SMALL      1.e-20

extern int analysis_flag, dof, integ_flag, numel, numnp, sof, flag_quad_element;
extern int LU_decomp_flag, numel_K, numel_P;
extern SH shg, shg_node, shl, shl_node, shl_tri;
extern ROTATE rotate, rotate_node;
extern double w[num_int8], w_tri[num_int6];
extern int  iteration_max, iteration_const, iteration;
extern double tolerance;

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int shellB1ptT(double *, double *,double *, double *, double *, double *);

int shellB1ptM(double *, double *, double *, double *);

int shellBNpt(double *, double *,double *, double *, double *, double *);

int shshg( double *, int , SH , SH , XL , double *, double *, double *,
	double *, ROTATE );

int dotX(double *, double *, double *, int);

int shBoundary( double *, BOUND );


int shConjPassemble(double *A, int *connect, double *coord, int *el_matl, double *lamina_ref,
	double *fiber_xyz, MATL *matl, double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 10/10/08
*/
	int i, i1, i2, i4, j, k, dof_el[neqel20], sdof_el[npel8*nsd];
	int check, node;
	int matl_num;
	double Emod, Pois, G1, G2, G3, shearK, const1, const2;
	double fdum, fdum1, fdum2, fdum3, fdum4;
	double D11,D12,D21,D22;
	double B[soB100], DB[soB100];
	double K_temp[neqlsq400], K_el[neqlsq400];
	double U_el[neqel20];
	double coord_el_trans[npel8*nsd], zm1[npell4], zp1[npell4],
		znode[npell4*num_ints], dzdt_node[npell4];
	double local_x[nsd], local_y[nsd], local_z[nsd], vec_dum[nsd];
	double det[num_int8+num_ints], wXdet;
	XL xl;
	double P_el[neqel20];
	int npell_, neqel_, num_intb_, neqlsq_, soB_, npel_;
	SH *shl_;

	npell_ = npell4;
	neqel_ = neqel20;
	num_intb_ = num_intb4;
	neqlsq_ = neqlsq400;
	soB_ = soB100;
	npel_ = npel8;
	shl_ = &shl;

	if(!flag_quad_element)
	{
		npell_ = npell3;
		neqel_ = neqel15;
		num_intb_ = num_intb3;
		neqlsq_ = neqlsq225;
		soB_ = soB75;
		npel_ = npel6;
		shl_ = &shl_tri;
	}

	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel_K; ++k )
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

		check = matX(P_el, (A+k*neqlsq_), U_el, neqel_, 1, neqel_);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel_; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	for( k = numel_K; k < numel; ++k )
	{
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		shearK = matl[matl_num].shear;

/* The constants below are for plane stress */

		const1 = Emod/(1.0-Pois*Pois);
		const2 = .5*(1.0-Pois);

		/*printf("Emod, Pois %f %f \n", Emod, Pois);*/

		D11 = const1;
		D12 = Pois*const1;
		D21 = Pois*const1;
		D22 = const1;

		G1 = const1*const2;
		G2 = const1*const2*shearK;
		G3 = const1*const2*shearK;

/* Create the coord transpose vector for one element */

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

		memcpy(rotate_node.f_shear,rotate.f_shear,sorfs36*sizeof(double));

/* Assembly of the shg matrix for each integration point */

		check=shshg( det, k, *shl_, shg, xl, zp1, zm1, znode,
			dzdt_node, rotate);
		if(!check) printf( "Problems with shshg \n");

		memset(U_el,0,neqel_*sof);
		memset(K_el,0,neqlsq_*sof);

/* The loop over i4 below calculates the 2 fiber points of the gaussian integration */
		for( i4 = 0; i4 < num_ints; ++i4 )
		{

/* The loop over j below calculates the 2X2 or 3 points of the gaussian integration
   over the lamina for several quantities */

		   for( j = 0; j < num_intb_; ++j )
		   {
			memset(B,0,soB_*sof);
			memset(DB,0,soB_*sof);
			memset(K_temp,0,neqlsq_*sof);

/* Assembly of the B matrix */

			check =
			   shellBNpt((shg.bend+npell_*(nsd+1)*num_intb_*i4+npell_*(nsd+1)*j),
			   (shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
			   (znode+npell_*i4),B,(rotate.l_bend+nsdsq*num_intb_*i4+nsdsq*j),
			   rotate.f_shear);
			if(!check) printf( "Problems with shellBNpt \n");

			if( integ_flag == 0 || integ_flag == 2 ) 
			{
/* Calculate the membrane shear terms in B using 1X1 point gaussian integration in lamina */

			    check = shellB1ptM((shg.shear+npell_*(nsd+1)*i4),
				(B+(sdim-2)*neqel_),
				(rotate.l_shear+nsdsq*i4), rotate.f_shear);
			    if(!check) printf( "Problems with shellB1pt \n");
			}

			if( integ_flag == 1 || integ_flag == 2 ) 
			{
/* Calculate the transverse shear terms in B using 1X1 point gaussian integration in lamina */

			    check = shellB1ptT((shg.shear+npell_*(nsd+1)*i4),
				(shg.bend_z+npell_*(nsd)*num_intb_*i4+npell_*(nsd)*j),
				(znode+npell_*i4),(B+(sdim-2)*neqel_),
				(rotate.l_shear+nsdsq*i4), rotate.f_shear);
			    if(!check) printf( "Problems with shellB1pt \n");
			}

			for( i1 = 0; i1 < neqel_; ++i1 )
			{
				*(DB+i1)=*(B+i1)*D11 +
					*(B+neqel_*1+i1)*D12;
				*(DB+neqel_*1+i1)=*(B+i1)*D21 +
					*(B+neqel_*1+i1)*D22;
				*(DB+neqel_*2+i1)=*(B+neqel_*2+i1)*G1;
				*(DB+neqel_*3+i1)=*(B+neqel_*3+i1)*G2;
				*(DB+neqel_*4+i1)=*(B+neqel_*4+i1)*G3;
			}

			wXdet = *(w+num_intb_*i4+j)*(*(det+num_intb_*i4+j));
			if(!flag_quad_element)
				wXdet = 0.5*(*(w_tri+num_intb_*i4+j))*(*(det+num_intb_*i4+j));

			check=matXT(K_temp, B, DB, neqel_, neqel_, sdim);
			if(!check) printf( "Problems with matXT \n");

			for( i2 = 0; i2 < neqlsq_; ++i2 )
			{
			   *(K_el+i2) += *(K_temp+i2)*wXdet;
			}
		   }
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel_; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, K_el, U_el, neqel_, 1, neqel_);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel_; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	return 1;
}


int shConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, double *lamina_ref, double *fiber_xyz,
	MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to shConjPassemble to get the
   product of [A]*[p].

                        Updated 10/10/06

   It is taken from the algorithm 10.3.1 given in "Matrix Computations",
   by Golub, page 534.
*/
	int i, j, sofmf, ptr_inc;
	int check, counter;
	double *mem_double;
	double *p, *P_global_CG, *r, *z;
	double alpha, alpha2, beta;
	double fdum, fdum2;

/* For the doubles */
	sofmf = 4*dof;
	mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
		printf( "failed to allocate memory for doubles\n ");
		exit(1);
	}

/* For the Conjugate Gradient Method doubles */

	                                        ptr_inc = 0;
	p=(mem_double+ptr_inc);                 ptr_inc += dof;
	P_global_CG=(mem_double+ptr_inc);       ptr_inc += dof;
	r=(mem_double+ptr_inc);                 ptr_inc += dof;
	z=(mem_double+ptr_inc);                 ptr_inc += dof;

/* Using Conjugate gradient method to find displacements */

	memset(P_global_CG,0,dof*sof);
	memset(p,0,dof*sof);
	memset(r,0,dof*sof);
	memset(z,0,dof*sof);

	for( j = 0; j < dof; ++j )
	{
		*(K_diag + j) += SMALL;
		*(r+j) = *(force+j);
		*(z + j) = *(r + j)/(*(K_diag + j));
		*(p+j) = *(z+j);
	}
	check = shBoundary (r, bc);
	if(!check) printf( " Problems with shBoundary \n");

	check = shBoundary (p, bc);
	if(!check) printf( " Problems with shBoundary \n");

	alpha = 0.0;
	alpha2 = 0.0;
	beta = 0.0;
	fdum2 = 1000.0;
	counter = 0;
	check = dotX(&fdum, r, z, dof);

	printf("\n iteration %3d iteration max %3d \n", iteration, iteration_max);
	/*for( iteration = 0; iteration < iteration_max; ++iteration )*/
	while(fdum2 > tolerance && counter < iteration_max )
	{

		printf( "\n %3d %16.8e\n",counter, fdum2);
		check = shConjPassemble( A, connect, coord, el_matl, lamina_ref,
			fiber_xyz, matl, P_global_CG, p);
		if(!check) printf( " Problems with shConjPassemble \n");
		check = shBoundary (P_global_CG, bc);
		if(!check) printf( " Problems with shBoundary \n");
		check = dotX(&alpha2, p, P_global_CG, dof);	
		alpha = fdum/(SMALL + alpha2);

		for( j = 0; j < dof; ++j )
		{
		    /*if( counter > 14 && counter < 18 )
		    printf( "%4d %18.10e  %16.8e  %16.8e  %16.8e  %18.10e %16.8e\n",j,alpha,
			beta,*(U+j),*(r+j),*(P_global_CG+j),*(p+j));*/
		    *(U+j) += alpha*(*(p+j));
		    *(r+j) -=  alpha*(*(P_global_CG+j));
		    *(z + j) = *(r + j)/(*(K_diag + j));
		}

		check = dotX(&fdum2, r, z, dof);
		beta = fdum2/(SMALL + fdum);
		fdum = fdum2;
		
		for( j = 0; j < dof; ++j )
		{
		    /*printf("\n  %3d %12.7f  %14.5f ",j,*(U+j),*(P_global_CG+j));*/
		    /*printf( "%4d %14.5f  %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
		    printf( "%4d %14.8f  %14.8f  %14.8f  %14.8f %14.8f\n",j,
			*(U+j)*beta,*(r+j)*beta,*(P_global_CG+j)*alpha,
			*(force+j)*alpha);*/
		    *(p+j) = *(z+j)+beta*(*(p+j));
		}
		check = shBoundary (p, bc);
		if(!check) printf( " Problems with shBoundary \n");

		++counter;
	}

	if(counter > iteration_max - 1 )
	{
		printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",counter,fdum2);
		printf( "Problem may not have converged during Conj. Grad.\n");
	}
/*
The lines below are for testing the quality of the calculation:

1) r should be 0.0
2) P_global_CG( = A*U ) - force should be 0.0
*/

/*
	check = shConjPassemble( A, connect, coord, el_matl, lamina_ref,
		fiber_xyz, matl, P_global_CG, U);
	if(!check) printf( " Problems with shConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}



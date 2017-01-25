/*
    This utility code uses the conjugate gradient method
    to solve the linear system [K][U] = [f] for a finite
    element program which does analysis on a thermal brick.  The first
    function assembles the P matrix caluclation.  It is called by
    the function "brTConjGrad" which allocates the memory and
    goes through the steps of the algorithm.  These go with the
    calculation of the temperature distribution.  The second function
    also assembles the P matrix and is called by "br2ConjGrad".
    These go with the calculation of the displacement.

    The equations for below can be seen in the ANSYS manual:

      Kohneke, Peter, *ANSYS User's Manual for Revision 5.0,
         Vol IV Theory, Swanson Analysis Systems Inc., 1992.

	        Updated 11/2/06

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
#include "../brick/brconst.h"
#include "br2struct.h"

#define SMALL      1.e-20

extern int analysis_flag, dof, Tdof, numel, numel_film, numnp, sof;
extern int LU_decomp_flag, numel_K, Tnumel_K, TBnumel_K, numel_P, Tnumel_P;
extern double shg[sosh], shl[sosh], shl_film[sosh_film], w[num_int], *Vol0;
extern int  iteration_max, iteration_const, iteration;
extern double tolerance; 

int brshface( double *, int , double *, double *);

int matX(double *,double *,double *, int ,int ,int );

int matXT(double *, double *, double *, int, int, int);

int brickB(double *,double *);

int brickB_T2(double *,double *);

int brickB_T(double *,double *);

int brshg( double *, int, double *, double *, double *);

int dotX(double *, double *, double *, int);

int Boundary( double *, BOUND );

int br2Boundary( double *, BOUND );

int brTConjPassemble(double *A, int *connect, int *connect_film, double *coord,
	int *el_matl, int *el_matl_film, MATL *matl, double *P_global_CG, double *T)
{
/* This function assembles the P_global_CG matrix for the temperature calculation
   by taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

	                Updated 11/2/06
*/

	int i, i1, i2, j, k, Tdof_el[Tneqel], TBdof_el[TBneqel],
		sdof_el[npel*nsd];
	int check, node;
	int matl_num, matl_num_film;
	XYZF thrml_cond;
	double film_const;
	double fdum, fdum2;
	double B_T[soB], B_TB[TBsoB], DB[soB];
	double K_temp[Tneqlsq], K_el[Tneqlsq];
	double T_el[Tneqel], TB_el[Tneqel];
	double coord_el_trans[npel*nsd];
	double det[num_int], dArea[num_int_film], wXdet, wXdArea;
	double P_el[Tneqel], PB_el[TBneqel];


	memset(P_global_CG,0,Tdof*sof);

/* This loop uses the pre-assembled element stiffness matrices to find
   P_global_CG */

	for( k = 0; k < Tnumel_K; ++k )
	{

		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(Tdof_el+Tndof*j) = Tndof*node;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < Tneqel; ++j )
		{
			*(T_el + j) = *(T + *(Tdof_el+j));
		}

		check = matX(P_el, (A+k*Tneqlsq), T_el, Tneqel, 1, Tneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < Tneqel; ++j )
		{
			*(P_global_CG+*(Tdof_el+j)) += *(P_el+j);
		}
	}

/* This loop re-calculates the remaining element stiffness matrices
before calculating P_global_CG */

	for( k = Tnumel_K; k < numel; ++k )
	{
		matl_num = *(el_matl+k);

		thrml_cond.x = matl[matl_num].thrml_cond.x;
		thrml_cond.y = matl[matl_num].thrml_cond.y;
		thrml_cond.z = matl[matl_num].thrml_cond.z;

     /*printf("thrml K x, K y, K z %f %f %f \n",
		thrml_cond.x, thrml_cond.y, thrml_cond.z );*/

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

			*(Tdof_el+Tndof*j) = Tndof*node;
		}


/* Assembly of the shg matrix for each integration point */

		check = brshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* The loop over j below calculates the 8 points of the gaussian integration 
   for several quantities */

		memset(K_el,0,Tneqlsq*sof);
		memset(T_el,0,Tneqel*sof);

		for( j = 0; j < num_int; ++j )
		{
		    memset(B_T,0,TsoB*sof);
		    memset(DB,0,TsoB*sof);
		    memset(K_temp,0,Tneqlsq*sof);

/* Assembly of the B matrix */

		    check = brickB_T((shg+npel*(nsd+1)*j),B_T);
		    if(!check) printf( "Problems with brickB_T \n");

		    for( i1 = 0; i1 < Tneqel; ++i1 )
		    {
			*(DB+i1) = *(B_T+i1)*thrml_cond.x;
			*(DB+Tneqel*1+i1) = *(B_T+Tneqel*1+i1)*thrml_cond.y;
			*(DB+Tneqel*2+i1) = *(B_T+Tneqel*2+i1)*thrml_cond.z;
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check = matXT(K_temp, B_T, DB, Tneqel, Tneqel, Tdim);
		    if(!check) printf( "error \n");

/* Compute the element diffusion conductivity matrix.  Look at the [Ktb] matrix
   on page 6-6 of the ANSYS manual.
*/
		    for( i2 = 0; i2 < Tneqlsq; ++i2 )
		    {
			  *(K_el+i2) += *(K_temp+i2)*wXdet;
		    }
		}

/* Assembly of the global P matrix */

		for( j = 0; j < Tneqel; ++j )
		{
			*(T_el + j) = *(T + *(Tdof_el+j));
		}

		check = matX(P_el, K_el, T_el, Tneqel, 1, Tneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < Tneqel; ++j )
		{
			*(P_global_CG+*(Tdof_el+j)) += *(P_el+j);
		}
	}

/* This loop uses the pre-assembled surface element stiffness matrices to find
   P_global_CG */

	if(TBnumel_K)
	{
	    for( k = 0; k < numel_film; ++k )
	    {

		for( j = 0; j < npel_film; ++j )
		{
			node = *(connect_film+npel_film*k+j);

			*(TBdof_el+Tndof*j) = Tndof*node;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < TBneqel; ++j )
		{
			*(TB_el + j) = *(T + *(TBdof_el+j));  /* This is correct, T not TB */
		}

		check = matX(P_el, (A + Tnumel_K*Tneqlsq + k*TBneqlsq),
			TB_el, TBneqel, 1, TBneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < TBneqel; ++j )
		{
			*(P_global_CG+*(TBdof_el+j)) += *(P_el+j);
		}
	    }
	}
	else
	{
	    for( k = 0; k < numel_film; ++k )
	    {
		matl_num_film = *(el_matl_film+k);
		film_const = matl[matl_num_film].film;

/* Create the coord_el transpose vector for one element */

		for( j = 0; j < npel_film; ++j )
		{
			node = *(connect_film+npel_film*k+j);

			*(sdof_el+nsd*j) = nsd*node;
			*(sdof_el+nsd*j+1) = nsd*node+1;
			*(sdof_el+nsd*j+2) = nsd*node+2;

			*(coord_el_trans+j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el_trans+npel_film*1+j)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el_trans+npel_film*2+j)=*(coord+*(sdof_el+nsd*j+2));

			*(TBdof_el+Tndof*j) = Tndof*node;
		}

		memset(TB_el,0,TBneqel*sof);

		for( j = 0; j < TBneqel; ++j )
		{
			*(TB_el + j) = *(T + *(TBdof_el+j));  /* This is correct, T not TB */
		}

		check = brshface( dArea, k, shl_film, coord_el_trans);
		if(!check) printf( "Problems with brshface \n");

		memset(K_el,0,TBneqlsq*sof);

		for( j = 0; j < num_int_film; ++j )
		{
		   memset(B_TB,0,TBneqel*sof);
		   memset(DB,0,TBneqel*sof);
		   memset(K_temp,0,TBneqlsq*sof);

		   check = brickB_T2((shl_film+npel_film*nsd*j+npel_film*2),B_TB);
		   if(!check) printf( "Problems with brickB_T2 \n");

		   *(DB) = *(B_TB)*film_const;
		   *(DB+1) = *(B_TB+1)*film_const;
		   *(DB+2) = *(B_TB+2)*film_const;
		   *(DB+3) = *(B_TB+3)*film_const;

		   wXdArea = *(w+j)*(*(dArea+j));

		   check = matXT(K_temp, B_TB, DB, TBneqel, TBneqel, 1);
		   if(!check) printf( "error \n");

/* Compute the element convection surface conductivity matrix.  Look at the [Ktc] matrix
   on page 6-6 of the ANSYS manual.
*/
		   for( i2 = 0; i2 < TBneqlsq; ++i2 )
		   {
		       *(K_el+i2) += *(K_temp+i2)*wXdArea;
		   }
		}

/* Assembly of the global P matrix */

		check = matX(P_el, K_el, TB_el, TBneqel, 1, TBneqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < TBneqel; ++j )
		{
		    *(P_global_CG+*(TBdof_el+j)) += *(P_el+j);
		}
	    }
	}

	return 1;
}

int br2ConjPassemble(double *A, int *connect, double *coord, int *el_matl, MATL *matl,
	double *P_global_CG, double *U)
{
/* This function assembles the P_global_CG matrix for the displacement calculation by
   taking the product [K_el]*[U_el].  Some of the [K_el] is stored in [A].

                        Updated 9/25/01
*/
	int i, i1, i2, j, k, dof_el[neqel], sdof_el[npel*nsd];
	int check, node;
	int matl_num;
	XYZF Emod, alpha;
	ORTHO Pois, G;
	double fdum, fdum2, h_const, Exh, Eyh, Ezh, ExEy, EyEz, ExEz;
	double D11,D12,D13,D21,D22,D23,D31,D32,D33;
	double lamda, mu;
	double B[soB], DB[soB];
	double K_temp[neqlsq], K_el[neqlsq];
	double U_el[neqel];
	double coord_el_trans[npel*nsd], 
		yzsq, zxsq, xzsq, xysq, xxyy, xzyz, xzxy, yzxy;
	double det[num_int], wXdet;
	double P_el[neqel];


	memset(P_global_CG,0,dof*sof);

	for( k = 0; k < numel_K; ++k )
	{
		for( j = 0; j < npel; ++j )
		{
			node = *(connect+npel*k+j);

			*(dof_el+ndof*j) = ndof*node;
			*(dof_el+ndof*j+1) = ndof*node+1;
			*(dof_el+ndof*j+2) = ndof*node+2;
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, (A+k*neqlsq), U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	for( k = numel_K; k < numel; ++k )
	{

		matl_num = *(el_matl+k);

		Emod.x = matl[matl_num].E.x;
		Emod.y = matl[matl_num].E.y;
		Emod.z = matl[matl_num].E.z;

		Pois.xy = matl[matl_num].nu.xy;
		Pois.xz = matl[matl_num].nu.xz;
		Pois.yz = matl[matl_num].nu.yz;

		Pois.xy = Pois.xy*Emod.y/Emod.x;
		Pois.zx = Pois.xz*Emod.z/Emod.x;
		Pois.yz = Pois.yz*Emod.z/Emod.y;

		alpha.x = matl[matl_num].thrml_expn.x;
		alpha.y = matl[matl_num].thrml_expn.y;
		alpha.z = matl[matl_num].thrml_expn.z;

		xysq = Pois.xy*Pois.xy;
		xzsq = Pois.xz*Pois.xz;
		yzsq = Pois.yz*Pois.yz;

		xzyz = Pois.xz*Pois.yz;
		xzxy = Pois.xz*Pois.xy;
		yzxy = Pois.yz*Pois.xy;

		G.xy = Emod.x*Emod.y/(Emod.x + Emod.y + 2*Pois.xy*Emod.x);
		G.xz = G.xy;
		G.yz = G.xy;

		h_const = 1.0 - xysq*Emod.x/Emod.y - yzsq*Emod.y/Emod.z -
			xzsq*Emod.x/Emod.z - 2*Pois.xy*Pois.yz*Pois.xz*Emod.x/Emod.z;

		Exh = Emod.x/h_const;
		Eyh = Emod.y/h_const;
		Ezh = Emod.z/h_const;

		ExEy = Emod.x/Emod.y;
		ExEz = Emod.x/Emod.z;
		EyEz = Emod.y/Emod.z;

		lamda = Emod.x*Pois.xy/((1.0+Pois.xy)*(1.0-2.0*Pois.xy));
		mu = Emod.x/(1.0+Pois.xy)/2.0;

/* Look at equations 2.1-18 to 2.1-20 on page 2-4 of the ANSYS manual to see how the
   D's below are defined.  This D is different from the [D] of equation 2.1-4 to 2.1-5.
*/

		D11 = Exh*(1.0 - yzsq*EyEz);
		D12 = Exh*(Pois.xy + xzyz*EyEz);
		D13 = Exh*(Pois.xz + yzxy);
		D21 = Exh*(Pois.xy + xzyz*EyEz);
		D22 = Eyh*(1.0 - xzsq*ExEz);
		D23 = Eyh*(Pois.yz + xzxy);
		D31 = Exh*(Pois.xz + yzxy);
		D32 = Eyh*(Pois.yz + xzxy);
		D33 = Eyh*(1.0 - xysq*ExEy);

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

/* Assembly of the shg matrix for each integration point */

		memset(shg,0,sosh*sof);
		check = brshg(det, k, shl, shg, coord_el_trans);
		if(!check) printf( "Problems with brshg \n");

/* The loop over j below calculates the 8 points of the gaussian integration 
   for several quantities */

		memset(U_el,0,neqel*sof);
		memset(K_el,0,neqlsq*sof);

		for( j = 0; j < num_int; ++j )
		{

		    memset(B,0,soB*sof);
		    memset(DB,0,soB*sof);
		    memset(K_temp,0,neqlsq*sof);

/* Assembly of the B matrix */

		    check = brickB((shg+npel*(nsd+1)*j),B);
		    if(!check) printf( "Problems with brickB \n");

		    for( i1 = 0; i1 < neqel; ++i1 )
		    {
			*(DB+i1) = *(B+i1)*D11+
				*(B+neqel*1+i1)*D12+
				*(B+neqel*2+i1)*D13;
			*(DB+neqel*1+i1) = *(B+i1)*D21+
				*(B+neqel*1+i1)*D22+
				*(B+neqel*2+i1)*D23;
			*(DB+neqel*2+i1) = *(B+i1)*D31+
				*(B+neqel*1+i1)*D32+
				*(B+neqel*2+i1)*D33;
			*(DB+neqel*3+i1) = *(B+neqel*3+i1)*G.xy;
			*(DB+neqel*4+i1) = *(B+neqel*4+i1)*G.xz;
			*(DB+neqel*5+i1) = *(B+neqel*5+i1)*G.yz; 
		    }

		    wXdet = *(w+j)*(*(det+j));

		    check = matXT(K_temp, B, DB, neqel, neqel, sdim);
		    if(!check) printf( "Problems with matXT \n");
		    for( i2 = 0; i2 < neqlsq; ++i2 )
		    {
			  *(K_el+i2) += *(K_temp+i2)*wXdet;
		    }
		}

/* Assembly of the global P matrix */

		for( j = 0; j < neqel; ++j )
		{
			*(U_el + j) = *(U + *(dof_el+j));
		}

		check = matX(P_el, K_el, U_el, neqel, 1, neqel);
		if(!check) printf( "Problems with matX \n");

		for( j = 0; j < neqel; ++j )
		{
			*(P_global_CG+*(dof_el+j)) += *(P_el+j);
		}
	}

	return 1;
}

int brTConjGrad(double *A, BOUND bc, int *connect, int *connect_film, double *coord,
	int *el_matl, int *el_matl_film, double *Q, MATL *matl, double *T,
	double *TK_diag)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   temperature.  It also makes the call to brTConjPassemble to get the
   product of [A]*[p].

                        Updated 1/7/03

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
	sofmf = 4*Tdof;
	mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
		printf( "failed to allocate memory for doubles\n ");
		exit(1);
	}

/* For the Conjugate Gradient Method doubles */

	                                        ptr_inc = 0;
	p=(mem_double+ptr_inc);                 ptr_inc += Tdof;
	P_global_CG=(mem_double+ptr_inc);       ptr_inc += Tdof;
	r=(mem_double+ptr_inc);                 ptr_inc += Tdof;
	z=(mem_double+ptr_inc);                 ptr_inc += Tdof;

/* Using Conjugate gradient method to find temperature distribution */

	memset(P_global_CG,0,Tdof*sof);
	memset(p,0,Tdof*sof);
	memset(r,0,Tdof*sof);
	memset(z,0,Tdof*sof);

	for( j = 0; j < Tdof; ++j )
	{
		*(TK_diag + j) += SMALL;
		*(r+j) = *(Q+j);
		*(z + j) = *(r + j)/(*(TK_diag + j));
		*(p+j) = *(z+j);
/*
		 *(r+j) = *(T+j);
		 *(p+j) = *(T+j);

		 fdum += 3.14159/((double)Tdof);
		 *(r+j) = sin(fdum);
		 *(p+j) = sin(fdum);
*/
	}
	check = br2Boundary (r, bc);
	if(!check) printf( " Problems with br2Boundary \n");

	check = br2Boundary (p, bc);
	if(!check) printf( " Problems with br2Boundary \n");

	alpha = 0.0;
	alpha2 = 0.0;
	beta = 0.0;
	fdum2 = 1000.0;
	counter = 0;
	check = dotX(&fdum, r, z, Tdof);

	printf("\n iteration %3d iteration max %3d \n", iteration, iteration_max);
	/*for( iteration = 0; iteration < iteration_max; ++iteration )*/
	while(fdum2 > tolerance && counter < iteration_max)
	{
		printf( "\n %3d %14.6e \n",counter, fdum2);
		check = brTConjPassemble( A, connect, connect_film, coord, el_matl,
			el_matl_film, matl, P_global_CG, p);
		if(!check) printf( " Problems with brTConjPassemble \n");
		check = br2Boundary (P_global_CG, bc);
		if(!check) printf( " Problems with br2Boundary \n");
		check = dotX(&alpha2, p, P_global_CG, Tdof);	
		alpha = fdum/(SMALL + alpha2);
		for( j = 0; j < Tdof; ++j )
		{
		    /*printf( "%4d %14.5e  %14.5e  %14.5e  %14.5e  %14.5e %14.5e\n",j,alpha,
				beta,*(T+j),*(r+j),*(P_global_CG+j),*(p+j));*/
		    *(T+j) += alpha*(*(p+j));
		    *(r+j) -=  alpha*(*(P_global_CG+j));
		    *(z + j) = *(r + j)/(*(TK_diag + j));
		}

		check = dotX(&fdum2, r, z, Tdof);
		beta = fdum2/(SMALL + fdum);
		fdum = fdum2;
	
		for( j = 0; j < Tdof; ++j )
		{
		    /*printf("\n  %3d %12.7f  %14.5f ",j,*(T+j),*(P_global_CG+j));*/
		    *(p+j) = *(z+j)+beta*(*(p+j));
		}
		check = br2Boundary (p, bc);
		if(!check) printf( " Problems with br2Boundary \n");

		++counter;
	}
	if(counter > iteration_max - 1 )
	{
		printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",
			counter,fdum2);
		printf( "Problem may not have converged during Conj. Grad. for temp.\n");
	}
/*
The lines below are for testing the quality of the calculation:

1) r should be 0.0
2) P_global_CG( = A*T ) - Q should be 0.0
*/

/*
	check = brTConjPassemble( A, connect, connect_film, coord, el_matl,
		el_matl_film, matl, P_global_CG, T);
	if(!check) printf( " Problems with brTConjPassemble \n");
*/

	free(mem_double);

	return 1;
}

int br2ConjGrad(double *A, BOUND bc, int *connect, double *coord, int *el_matl,
	double *force, double *K_diag, MATL *matl, double *U)
{
/* This function does memory allocation and uses the conjugate gradient
   method to solve the linear system arising from the calculation of
   displacements.  It also makes the call to br2ConjPassemble to get the
   product of [A]*[p].

                        Updated 1/24/06

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
	check = Boundary (r, bc);
	if(!check) printf( " Problems with Boundary \n");

	check = Boundary (p, bc);
	if(!check) printf( " Problems with Boundary \n");

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
		check = br2ConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, p);
		if(!check) printf( " Problems with br2ConjPassemble \n");
		check = Boundary (P_global_CG, bc);
		if(!check) printf( " Problems with Boundary \n");
		check = dotX(&alpha2, p, P_global_CG, dof);	
		alpha = fdum/(SMALL + alpha2);

		for( j = 0; j < dof; ++j )
		{
		    /*printf( "%4d %14.5e  %14.5e  %14.5e  %14.5e  %14.5e %14.5e\n",j,alpha,
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
		    *(p+j) = *(z+j)+beta*(*(p+j));
		}
		check = Boundary (p, bc);
		if(!check) printf( " Problems with Boundary \n");

		++counter;
	}

	if(counter > iteration_max - 1 )
	{
		printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",counter,fdum2);
		printf( "Problem may not have converged during Conj. Grad. for disp.\n");
	}
/*
The lines below are for testing the quality of the calculation:

1) r should be 0.0
2) P_global_CG( = A*U ) - force should be 0.0
*/

/*
	check = br2ConjPassemble( A, connect, coord, el_matl, matl, P_global_CG, U);
	if(!check) printf( " Problems with br2ConjPassemble \n");

	for( j = 0; j < dof; ++j )
	{
		printf( "%4d %14.5f  %14.5f %14.5f  %14.5f  %14.5f %14.5f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(force+j));
	}
*/

	free(mem_double);

	return 1;
}


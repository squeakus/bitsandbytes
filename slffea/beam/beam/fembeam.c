/*
    This program performs finite element analysis on 
    a beam by reading in the data, doing assembling, and
    then solving the linear system

	        Updated 11/4/09

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2009  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../../common/eigen.h"
#if B2D
#include "bmconst2D.h"
#include "bmstruct2D.h"
#endif
#if B3D
#include "bmconst.h"
#include "bmstruct.h"
#endif

int bmwriter ( double *, BOUND , int *, double *, CURVATURE *, int *, int *,
	double *, int *, MATL *, MOMENT *, char *, STRAIN *, STRESS *, double *);

int eval_data_print( EIGEN *, char *, int );

int bmLanczos(double *, BOUND , int *, double *, EIGEN *, int *, int *, int *,
	int *, double *, double *, double *, double *, MATL *, double *, int );

int bmConjGrad(double *, BOUND , int *, double *, int *, int *, double *,
	double *, double *, double *, MATL *, double *);

int solve(double *,double *,int *, int );

int decomp(double *,int *, int );

int bmMassemble( int *, double *, int *, int *, int *, double *, double *,
	double *, MATL *);

int bmKassemble(double *, BOUND , int *, double *, CURVATURE *, double *, int *,
	int *, double *, int *, int *, double *, int *, double *, double *, MATL *,
	MOMENT *, STRAIN *, STRESS *, double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int bmformid( BOUND, int *);

int bmsort(int *, int );

int bmlocal_vectors( double *, int *, double *, double *, double * );

int bmreader(double *, BOUND , int *, double *, double *, int *, int *, double *,
	MATL *, MOMENT *, FILE *, STRESS *, double *);

int bmMemory( double **, int , int **, int , MATL **, int , XYZPhiI **, int ,
	CURVATURE **, MOMENT **, STRAIN **, STRESS **, int  );

int analysis_flag, dof, sdof, modal_flag, neqn, nmat, nmode, numel, numnp, sof;
int static_flag, consistent_mass_flag, consistent_mass_store, eigen_print_flag,
	lumped_mass_flag, stress_read_flag, gauss_stress_flag, stress_xyzx_flag;

int LU_decomp_flag, numel_K, numel_P, numnp_LUD_max;
int iteration_max, iteration_const, iteration;
double tolerance;

double x[num_int], w[num_int];
double x1[num_int1], w1[num_int1], x3[num_int3], w3[num_int3],
	x4[num_int4], w4[num_int4], x_node[num_int];

int main(int argc, char** argv)
{
	int i, j;
	int *id, *lm, *idiag, check, name_length, counter, MemoryCounter;
	XYZPhiI *mem_XYZPhiI;
	int *mem_int, sofmA, sofmA_K, sofmA_mass, sofmi, sofmf, sofmSTRESS,
		sofmXYZPhiI, ptr_inc;
	MATL *matl; 
	double *mem_double;
	double fpointx, fpointy, fpointz;
	int *connect, *el_matl, *el_type, dum;
	double *coord, *dist_load, *force, *mass, *axis_z,
		*U, *A, *vector_dum, *local_xyz, *length;
	double *K_diag, *ritz;
	EIGEN *eigen;
	int num_eigen;
	char name[30], buf[ BUFSIZ ];
	char name_mode[30], *ccheck;
	FILE *o1, *o2;
	BOUND bc;
	MOMENT *moment;
	STRESS *stress;
	CURVATURE *curve;
	STRAIN *strain;
	double fdum;
	long timec;
	long timef;
	int  mem_case, mem_case_mass;
	double RAM_max, RAM_usable, RAM_needed, MEGS;

	sof = sizeof(double);

/* Create the nodal points */

	*(x_node)= -1.0;
	*(x_node+1)= 1.0;

/* Create the Gauss integration points and weights */

/* 1 pt. Gauss */

	*(x1)= 0.0;

	*(w1)= 2.0;

/* 2 pt. Gauss */

	*(x)= -1.0/sq3;
	*(x+1)= 1.0/sq3;

	*(w)= 1.0;
	*(w+1)= 1.0;

/* 3 pt. Gauss */

	fdum = 3.0/5.0;
	*(x3) = -sqrt(fdum);
	*(x3+1) =  0.0;
	*(x3+2) =  sqrt(fdum);

	*(w3)= 5.0/9.0;
	*(w3+1)= 8.0/9.0;
	*(w3+2)= 5.0/9.0;

/* 4 pt. Gauss */

	fdum = 1/7.0*(3.0 - 4.0*sqpt3);
	*(x4) = -sqrt(fdum);
	*(x4+2) =  sqrt(fdum);

	fdum = 1/7.0*(3.0 + 4.0*sqpt3);
	*(x4+1)= -sqrt(fdum);
	*(x4+3)= sqrt(fdum);

	*(w4)= .5 + 1.0/12.0*sq3pt33;
	*(w4+1)= .5 - 1.0/12.0*sq3pt33;
	*(w4+2)= .5 + 1.0/12.0*sq3pt33;
	*(w4+3)= .5 - 1.0/12.0*sq3pt33;


	memset(name,0,30*sizeof(char));

	printf("What is the name of the file containing the \n");
	printf("beam structural data? (example: bridge)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural data */
/*   o2 contains input parameters */

	o1 = fopen( name,"r" );
	o2 = fopen( "bminput","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file bminput\n");
		tolerance = 1.e-13;
		iteration_max = 2000;
		RAM_max = 160.0;
		gauss_stress_flag = 0;
		eigen_print_flag = 0;
	}
	else
	{
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%lf\n ",&tolerance);
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%d\n ",&iteration_max);
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%lf\n ",&RAM_max);
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%d\n ",&gauss_stress_flag);
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%d\n ",&eigen_print_flag);
	}

	fgets( buf, BUFSIZ, o1 );
	fscanf( o1, "%d %d %d %d\n ",&numel,&numnp,&nmat,&nmode);
	dof=numnp*ndof;
	sdof=numnp*nsd;

	numnp_LUD_max = 500;

/* Assuming Conjugate gradient method is used, determine how much RAM is needed.
   This determines the largest problem that can be run on this machine.
   If problem small enough that LU decomposition is used, then calculation below
   is irrelevant.

   RAM variables given in bytes

*/
	RAM_max *= MB;
	RAM_usable = 0.5*RAM_max;
	RAM_needed = numel*neqlsq*sof;

	fdum = RAM_usable - RAM_needed;
	if(fdum > 0.0)
	{
/* Enough RAM to store all element stiffness matrices for conjugate gradient*/
		numel_K = numel;
		numel_P = 0;
	}
	else
	{
/* Store numel_K element stiffness matrices while remaining numel_P element stiffness
   matrices are calculated through product [K_el][U_el] = [P_el] */

		numel_P = numel - (((int)RAM_usable)/((double)neqlsq*sof));
		numel_K = numel - numel_P;
	}

	LU_decomp_flag = 1;
	if(numnp > numnp_LUD_max) LU_decomp_flag = 0;

	static_flag = 1;
	modal_flag = 0;

	lumped_mass_flag = 1;
	consistent_mass_flag = 0;
	if(nmode < 0)
	{
		lumped_mass_flag = 0;
		consistent_mass_flag = 1;
		nmode = abs(nmode);
	}

	if(nmode)
	{

/* The criteria for over-calculating the number of desired eigenvalues is
   taken from "The Finite Element Method" by Thomas Hughes, page 578.  It
   is actually given for subspace iteration, but I find it works well
   for the Lanczos Method.  I have also modified the factors
   given in Hughes from 2.0 to 3.0 and 8 to 10.
*/

		num_eigen = (int)(3.0*nmode);
		num_eigen = MIN(nmode + 10, num_eigen);
		num_eigen = MIN(dof, num_eigen);
		static_flag = 0;
		modal_flag = 1;
	}

#if 0
	LU_decomp_flag = 0;
#endif

/*   Begin allocation of meomory */

	MemoryCounter = 0;

/* For the doubles */
	sofmf=sdof + numel*(nsd-1) + 3*dof + numel*nsd + dof + numel*nsdsq + numel;
	if(modal_flag)
	{
	    sofmf = sdof + numel*(nsd-1) + 3*dof + numel*nsd + dof + numel*nsdsq +
		numel + num_eigen*dof;
	}
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi= numel*npel + 2*dof + numel*npel*ndof + 2*numel + numnp+1 + numel+1 + 2;
	MemoryCounter += sofmi*sizeof(int);
	printf( "\n Memory requrement for integers is %15d bytes\n",MemoryCounter);

/* For the XYZPhiI integers */
	sofmXYZPhiI=numnp+1+1;
	MemoryCounter += sofmXYZPhiI*sizeof(XYZPhiI);
	printf( "\n Memory requrement for XYZPhiI integers is %15d bytes\n",MemoryCounter);

/* For the STRESS */
	sofmSTRESS=numel;
	MemoryCounter += sofmSTRESS*sizeof(STRESS) + sofmSTRESS*sizeof(STRAIN) +
		sofmSTRESS*sizeof(MOMENT) + sofmSTRESS*sizeof(CURVATURE);
	printf( "\n Memory requrement for STRESS doubles is %15d bytes\n",MemoryCounter);

	check = bmMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat, &mem_XYZPhiI,
		sofmXYZPhiI, &curve, &moment, &strain, &stress, sofmSTRESS );
	if(!check) printf( " Problems with bmMemory \n");

/* For the doubles */
	                                        ptr_inc=0;
	coord=(mem_double+ptr_inc);             ptr_inc += sdof;
	dist_load=(mem_double+ptr_inc);         ptr_inc += numel*(nsd-1);
	vector_dum=(mem_double+ptr_inc);        ptr_inc += dof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	axis_z=(mem_double+ptr_inc);            ptr_inc += numel*nsd;
	K_diag=(mem_double+ptr_inc);            ptr_inc += dof;
	local_xyz=(mem_double+ptr_inc);         ptr_inc += numel*nsdsq;
	length=(mem_double+ptr_inc);            ptr_inc += numel;

/* If modal analysis is desired, allocate for mass and ritz vectors */

	if(modal_flag)
	{
	    ritz=(mem_double+ptr_inc);          ptr_inc += num_eigen*dof;
	}

/* For the integers */
	                                        ptr_inc = 0;
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npel;
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
	idiag=(mem_int+ptr_inc);                ptr_inc += dof;
	lm=(mem_int+ptr_inc);                   ptr_inc += numel*npel*ndof;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	el_type=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.dist_load=(mem_int+ptr_inc);         ptr_inc += numel+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;
	bc.num_dist_load=(mem_int+ptr_inc);     ptr_inc += 1;

/* For the XYZPhiI integers */
	                                     ptr_inc = 0;
	bc.fix =(mem_XYZPhiI+ptr_inc);       ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZPhiI+ptr_inc);    ptr_inc += 1;

/* If modal analysis is desired, allocate for the eigens */

	if(modal_flag)
	{
		eigen=(EIGEN *)calloc(num_eigen,sizeof(EIGEN));
		if(!eigen )
		{
			printf( "failed to allocate memory for eigen\n ");
			exit(1);
		}
	}

	timec = clock();
	timef = 0;

/* Initialize axis_z vectors */

	for( i = 0; i < numel; ++i )
	{
		*(axis_z + nsd*i + 2) = 1.0;
	}

	stress_read_flag = 1;
	check = bmreader(axis_z, bc, connect, coord, dist_load, el_matl, el_type,
		force, matl, moment, o1, stress, U);
	if(!check) printf( " Problems with bmreader \n");

	check = bmlocal_vectors( axis_z, connect, coord, length, local_xyz );
	if(!check) printf( " Problems with bmlocal_vectors \n");

/* Resort the variable bc.dist_load */

	if( bc.num_dist_load[0] > 1)
	{
		check = bmsort( bc.dist_load, bc.num_dist_load[0]);
		if(!check) printf( " Problems with bmsort \n");
	}

	printf(" \n\n");

	check = bmformid( bc, id );
	if(!check) printf( " Problems with bmformid \n");
/*
	printf( "\n This is the id matrix \n");
	for( i = 0; i < numnp; ++i )
	{
		printf("\n node(%4d)",i); 
		for( j = 0; j < ndof; ++j )
		{
			printf(" %4d  ",*(id+ndof*i+j));
		}
	}
*/
	check = formlm( connect, id, lm, ndof, npel, numel );
	if(!check) printf( " Problems with formlm \n");
/*
	printf( "\n\n This is the lm matrix \n");
	for( i = 0; i < numel; ++i )
	{
	    printf("\n element(%4d)",i);
	    for( j = 0; j < neqel; ++j )
	    {
		printf( "%5d   ",*(lm+neqel*i+j));
	    }
	}
	printf( "\n");
*/
	check = diag( idiag, lm, ndof, neqn, npel, numel);
	if(!check) printf( " Problems with diag \n");

/*
	printf( "\n\n This is the idiag matrix \n");
	for( i = 0; i < neqn; ++i )
	{
	    printf( "\ndof %5d   %5d",i,*(idiag+i));
	}
	printf( "\n");
*/

/*   allocate meomory for A, the global stiffness.  There are 2 possibilities:

     1) Use standard Linear Algebra with LU decomposition and skyline storage
	of global stiffness matrix.

     2) Use the Conjugate Gradient method with storage of numel_K element
	stiffness matrices.
*/
	sofmA = numel_K*neqlsq;                  /* case 2 */
	mem_case = 2;

	if(LU_decomp_flag)
	{
		sofmA = *(idiag+neqn-1)+1;       /* case 1 */
		mem_case = 1;
	}

	if( sofmA*sof > (int)RAM_usable )
	{

/* Even if the LU decomposition flag is on because there are only a few nodes, there
   is a possibility that there is not enough memory because of poor node numbering.
   If this is the case, then we have to use the conjugate gradient method.
 */

		sofmA = numel_K*neqlsq;
		LU_decomp_flag = 0;
		mem_case = 2;
	}

	printf( "\n We are in case %3d\n\n", mem_case);
	switch (mem_case) {
		case 1:
			printf( " LU decoomposition\n\n");
		break;
		case 2:
			printf( " Conjugate gradient \n\n");
		break;
	}

/* If modal analysis is desired, determine how much memory is needed and available
   for mass matrix */

	sofmA_K = sofmA;                                /* mass case 3 */
	sofmA_mass = 0;
	consistent_mass_store = 0;
	mem_case_mass = 3;

	if(modal_flag)
	{
	    if(consistent_mass_flag)
	    {
		fdum = RAM_usable - (double)(sof*(sofmA + numel*neqlsq));
		if(fdum > 0.0)
		{
/* Enough RAM to store all element mass matrices for modal analysis */

			sofmA_mass = numel*neqlsq;      /* mass case 2 */
			sofmA += sofmA_mass;
			consistent_mass_store = 1;
			mem_case_mass = 2;
		}
	    }
	    else
	    {
/* Using lumped mass so store only a diagonal mass matrix */

		sofmA_mass = dof;                      /* mass case 1 */
		sofmA += sofmA_mass;
		mem_case_mass = 1;
	    }

	    printf( "\n We are in mass case %3d\n\n", mem_case_mass);
	    switch (mem_case_mass) {
		case 1:
			printf( " Diagonal lumped mass matrix \n\n");
		break;
		case 2:
			printf( " Consistent mass matrix with all\n");
			printf( " element masses stored \n\n");
		break;
		case 3:
			printf( " Consistent mass matrix with no\n");
			printf( " storage of element masses \n\n");
		break;
	    }
	}

	MemoryCounter += sofmA*sizeof(double);
	printf( "\n Memory requrement for disp calc. with %15d doubles is %15d bytes\n",
		sofmA, MemoryCounter);
	MEGS = ((double)(MemoryCounter))/MB;
	printf( "\n Which is %16.4e MB\n", MEGS);
	printf( "\n This is numel, numel_K, numel_P %5d %5d %5d\n", numel, numel_K, numel_P);

#if 0
	consistent_mass_store = 0;
#endif

	if(sofmA < 1)
	{
		sofmA = 2;
		sofmA_K = 1;
	}

	A=(double *)calloc(sofmA,sizeof(double));
	if(!A )
	{
		printf( "failed to allocate memory for A double\n ");
		exit(1);
	}

/* Allocate memory for mass matrix */

	if(modal_flag)
	{
	                                       ptr_inc = sofmA_K;
		mass = (A + ptr_inc);          ptr_inc += sofmA_mass;
	}

	analysis_flag = 1;
	memset(A,0,sofmA*sof);
	check = bmKassemble(A, bc, connect, coord, curve, dist_load, el_matl,
		el_type, force, id, idiag, K_diag, lm, length, local_xyz,
		matl, moment, strain, stress, U);
	if(!check) printf( " Problems with bmKassemble \n");


/*	printf( "\n\n This is the force matrix \n");
	for( i = 0; i < neqn; ++i )
	{
	    printf( "\ndof %5d   %14.5f",i,*(force+i));
	}
	printf(" \n"); */


	if(modal_flag)
	{
	    if( lumped_mass_flag || consistent_mass_store )
	    {

/* For modal analysis, assemble either the diagonal lumped mass matrix or in
   the case of consistent mass, create and store all element mass matrices (if
   there is enough memory).
*/

/* Create mass matrix for modal analysis */

		check = bmMassemble( connect, coord, el_matl, el_type, id, length,
			local_xyz, mass, matl);
		if(!check) printf( " Problems with bmMassemble \n"); 

		/*if( lumped_mass_flag )
		{
		    printf( "\n\n This is the diagonal lumped mass matrix \n");
		    for( i = 0; i < neqn; ++i )
		    {
			printf( "\ndof %5d   %14.5f",i,*(mass+i));
		    }
		    printf(" \n");
		}*/
	    }
	}

	if(LU_decomp_flag)
	{

/* Perform LU Crout decompostion on the system */

		check = decomp(A,idiag,neqn);
		if(!check) printf( " Problems with decomp \n");
	}

/* If standard analysis desired, solve the system */

	if(static_flag)
	{
	    if(LU_decomp_flag)
	    {

/* Using LU decomposition to solve the system */

		check = solve(A,force,idiag,neqn);
		if(!check) printf( " Problems with solve \n");

		/* printf( "\n This is the solution to the problem \n");*/
		for( i = 0; i < dof; ++i )
		{
		    if( *(id + i) > -1 )
		    {
			*(U + i) = *(force + *(id + i));
		    }
		}

	    }

/* Using Conjugate gradient method to solve the system */

	    if(!LU_decomp_flag)
	    {
		check = bmConjGrad( A, bc, connect, coord, el_matl, el_type, force,
			K_diag, length, local_xyz, matl, U);
		if(!check) printf( " Problems with bmConjGrad \n");
	    }
/*
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) > -1 )
		{
			printf("\n node %3d x   %14.6e ",i,*(U+ndof*i));
		}
		if( *(id+ndof*i+1) > -1 )
		{
			printf("\n node %3d y   %14.6e ",i,*(U+ndof*i+1));
		}
		if( *(id+ndof*i+2) > -1 )
		{
			printf("\n node %3d z   %14.6e ",i,*(U+ndof*i+2));
		}
		if( *(id+ndof*i+3) > -1 )
		{
			printf("\n node %3d phi x   %14.6e ",i,*(U+ndof*i+3));
		}
		if( *(id+ndof*i+4) > -1 )
		{
			printf("\n node %3d phi y   %14.6e ",i,*(U+ndof*i+4));
		}
		if( *(id+ndof*i+5) > -1 )
		{
			printf("\n node %3d phi z   %14.6e ",i,*(U+ndof*i+5));
		}
	    }
	    printf(" \n");
*/

/* Calculate the reaction forces */
	    analysis_flag = 2;
	    memset(force,0,dof*sof);
	    /*printf( "\n\n These are the axial displacements and forces \n");*/
	    check = bmKassemble(A, bc, connect, coord, curve, dist_load, el_matl,
		el_type, force, id, idiag, K_diag, lm, length, local_xyz,
		matl, moment, strain, stress, U);
	    if(!check) printf( " Problems with bmKassemble \n");
	    check = dotX(&fdum, U, force, dof);
        printf( "\n\nThis is the strain energy: %16.8e ", fdum);
/*
	    printf( "\n\n These are the reaction forces \n");
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) < 0 )
		{
			printf("\n node %3d x   %14.6f ",i,*(force+ndof*i));
		}
		if( *(id+ndof*i+1) < 0 )
		{
			printf("\n node %3d y   %14.6f ",i,*(force+ndof*i+1));
		}
		if( *(id+ndof*i+2) < 0 )
		{
			printf("\n node %3d z   %14.6f ",i,*(force+ndof*i+2));
		}
		if( *(id+ndof*i+3) < 0 )
		{
			printf("\n node %3d z   %14.6f ",i,*(force+ndof*i+3));
		}
		if( *(id+ndof*i+4) < 0 )
		{
			printf("\n node %3d z   %14.6f ",i,*(force+ndof*i+4));
		}
		if( *(id+ndof*i+5) < 0 )
		{
			printf("\n node %3d z   %14.6f ",i,*(force+ndof*i+5));
		}
	    }

	    printf( "\n\n               These are the updated coordinates \n");
	    printf( "\n                  x               y             z \n");

	    for( i = 0; i < numnp; ++i )
	    {
		fpointx = *(coord+nsd*i) + *(U+ndof*i);
		fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1);
		fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
		printf("\n node %3d %14.9f %14.9f %14.9f",i,fpointx,fpointy,fpointz);
	    }
	    printf(" \n");
*/
	    check = bmwriter ( axis_z, bc, connect, coord, curve, el_matl,
		el_type, force, id, matl, moment, name, strain, stress, U);
	    if(!check) printf( " Problems with bmwriter \n");
	}

	counter = 0;
	if(modal_flag)
	{
	    name_length = strlen(name);
	    if( name_length > 20) name_length = 20;

	    memset(name_mode,0,30*sizeof(char));

/* name_mode is the name of the output data file for the mode shapes.
   It changes based on the number of the eigenmode.
*/
	    ccheck = strncpy(name_mode, name, name_length);
	    if(!ccheck) printf( " Problems with strncpy \n");

/* Number of calculated eigenvalues cannot exceed neqn */
	    num_eigen = MIN(neqn, num_eigen);
	    printf("\n The number of eigenvalues is: %5d \n", num_eigen);

/* nmode cannot exceed num_eigen */
	    nmode = MIN(num_eigen, nmode);

	    if(num_eigen < 2)
	    {
		printf("\n There is only one DOF \n");
		printf("\n so no modal analysis is performed.\n");
		exit(1);
	    }

/* Use Lanczos method for determining eigenvalues */

	    check = bmLanczos(A, bc, connect, coord, eigen, el_matl, el_type, id,
		idiag, K_diag, length, local_xyz, mass, matl, ritz, num_eigen);
	    if(!check) printf( " Problems with bmLanczos \n");

/* Write out the eigenvalues to a file */

	    check = eval_data_print( eigen, name, nmode);
	    if(!check) printf( " Problems with eval_data_print \n");

/* Write out the eigenmodes to a (name).mod-x.obm file */

	    for( j = 0; j < nmode; ++j )
	    {
		for( i = 0; i < dof; ++i )
		{
		    if( *(id + i) > -1 )
		    {
			*(U + i) = *(ritz + num_eigen*(*(id + i)) + j);
		    }
		}

		printf( "\n Eigenvalue %4d = %16.8e",j+1,eigen[j].val);

/* write the number of the eigenmode onto the name of the output file, name_mode */

		sprintf((name_mode+name_length+5), "%d",j+1);
		if(j + 1 > 9 )
			sprintf((name_mode+name_length+5), "%2d",j+1);
		if(j + 1 > 99 )
			sprintf((name_mode+name_length+5), "%3d",j+1);

		ccheck = strncpy(name_mode+name_length, ".mod-", 5);
		if(!ccheck) printf( " Problems with strncpy \n");

/* Re-initialize the stress, strain, moment, curvature */

		memset(moment,0,sofmSTRESS*sizeof(MOMENT));
		memset(stress,0,sofmSTRESS*sizeof(STRESS));
		memset(curve,0,sofmSTRESS*sizeof(CURVATURE));
		memset(strain,0,sofmSTRESS*sizeof(STRAIN));

/* Calculate the stresses */
		analysis_flag = 2;

		check = bmKassemble(A, bc, connect, coord, curve, dist_load, el_matl,
		    el_type, vector_dum, id, idiag, K_diag, lm, length, local_xyz,
		    matl, moment, strain, stress, U);
		if(!check) printf( " Problems with bmKassemble \n");

        

		check = bmwriter ( axis_z, bc, connect, coord, curve, el_matl,
		    el_type, force, id, matl, moment, name_mode, strain, stress,
		    U);
		if(!check) printf( " Problems with bmwriter \n");

		++counter;
	    }
	}
	
/*	printf( "\n\n\nbmKassemble? No idea what it is...\n\n\n", bmKassemble(A, bc, connect, coord, curve, dist_load, el_matl,
		    el_type, vector_dum, id, idiag, K_diag, lm, length, local_xyz,
		    matl, moment, strain, stress, U)); */
	
	timec = clock();
/*	printf("\n\n elapsed arse CPU = %lf\n\n",( (double)timec)/800.); */

	free(curve);
	free(moment);
	free(strain);
	free(stress);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZPhiI);
	if(modal_flag) free(eigen);
	free(A);

	return 1;
}

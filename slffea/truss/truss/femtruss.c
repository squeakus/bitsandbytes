/*
    This program performs finite element analysis
    by reading in the data, doing assembling, and
    then solving the linear system for a truss

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
#include "tsconst.h"
#include "tsstruct.h"

int tswriter ( BOUND , int *, double *, int *, double *, int *, MATL *,
	char *, SDIM *, SDIM *, double *);

int tsLength( int *, double *, double *);

int eval_data_print( EIGEN *, char *, int );

int tsLanczos(double *, BOUND , int *, double *, EIGEN *, int *, int *, int *,
	double *, double *, double *, double *, MATL *, double *, int );

int tsConjGrad(double *, BOUND , int *, double *, int *, double *, double *,
	double *, double *, MATL *, double *);

int solve(double *,double *,int *, int );

int decomp(double *, int *, int );

int tsMassemble( int *, double *, int *, int *, double *, double *, double *,
	MATL *);

int tsKassemble(double *, int *, double *, int *, double *, int *, int *,
	double *, int *lm, double *, double *, MATL *, SDIM *, SDIM *,
	double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int formid( BOUND, int *);

int tslocal_vectors( int *, double *, double *, double * );

int tsreader( BOUND , int *, double *, int *, double *, MATL *, 
	FILE *, SDIM *, double *);

int tsMemory( double **, int, int **, int, MATL **, int , XYZI **, int,
	SDIM **, SDIM **, int );

int dof, sdof, analysis_flag, modal_flag, neqn, nmat, nmode, numel, numnp, sof;
int static_flag, consistent_mass_flag, consistent_mass_store, eigen_print_flag,
	lumped_mass_flag, stress_read_flag, gauss_stress_flag;

int LU_decomp_flag, numel_K, numel_P, numnp_LUD_max;
int iteration_max, iteration_const, iteration;
double tolerance;

int main(int argc, char** argv)
{
	int i,j;
	int *id, *lm, *idiag, check, name_length, counter, MemoryCounter;
	XYZI *mem_XYZI;
	int *mem_int, sofmA, sofmA_K, sofmA_mass, sofmi, sofmf, sofmSDIM,
		sofmXYZI, ptr_inc;
	MATL *matl;
	double *mem_double;
	double fpointx, fpointy, fpointz;
	int *connect, *el_matl, dum;
	double *coord, *force, *mass, *U, *length, *lengthn, *A, *vector_dum,
		*local_xyz;
	double *K_diag, *ritz;
	EIGEN *eigen;
	int num_eigen;
	char name[30], buf[ BUFSIZ ];
	char name_mode[30], *ccheck;
	FILE *o1, *o2;
	BOUND bc;
	SDIM *stress;
	SDIM *strain;
	double fdum;
	long timec;
	long timef;
	int  mem_case, mem_case_mass;
	double RAM_max, RAM_usable, RAM_needed, MEGS;

	sof = sizeof(double);

	memset(name,0,30*sizeof(char));

	printf("What is the name of the file containing the \n");
	printf("truss structural data? (example: tower)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural data */
/*   o2 contains input parameters */

	o1 = fopen( name ,"r" );
	o2 = fopen( "tsinput","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file tsinput\n");
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

	numnp_LUD_max = 750;

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

/* The criteria for over-calculating the number of desired eigenvalues is
   taken from "The Finite Element Method" by Thomas Hughes, page 578.  It
   is actually given for subspace iteration, but I find it works well
   for the Lanczos Method.  I have also modified the factors
   given in Hughes from 2.0 to 5.0 and 8 to 16.
*/

	if(nmode)
	{
		num_eigen = (int)(2.8*nmode);
		num_eigen = MIN(num_eigen + 10, num_eigen);
		num_eigen = MIN(dof, num_eigen);
		static_flag = 0;
		modal_flag = 1;
	}
	num_eigen = 14;

#if 0
	LU_decomp_flag = 0;
#endif

/*   Begin allocation of meomory */

	MemoryCounter = 0;

/* For the doubles */
	sofmf = sdof + 3*dof + 2*numel + dof + numel*nsdsq;
	if(modal_flag)
	{
	    sofmf = sdof + 3*dof + 2*numel + dof + numel*nsdsq +
		num_eigen*dof;
	}
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi= numel*npel + 2*dof + numel*npel*ndof + numel + numnp+1 + 1;
	MemoryCounter += sofmi*sizeof(int);
	printf( "\n Memory requrement for integers is %15d bytes\n",MemoryCounter);

/* For the XYZI integers */
	sofmXYZI=numnp+1+1;
	MemoryCounter += sofmXYZI*sizeof(XYZI);
	printf( "\n Memory requrement for XYZI integers is %15d bytes\n",MemoryCounter);

/* For the SDIM */
	sofmSDIM=numel;
	MemoryCounter += sofmSDIM*sizeof(SDIM) + sofmSDIM*sizeof(SDIM);
	printf( "\n Memory requrement for SDIM doubles is %15d bytes\n",MemoryCounter);

	check = tsMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZI, sofmXYZI, &strain, &stress, sofmSDIM );
	if(!check) printf( " Problems with tsMemory \n");

/* For the doubles */
	                                        ptr_inc = 0;
	coord=(mem_double+ptr_inc);             ptr_inc += sdof;
	vector_dum=(mem_double+ptr_inc);        ptr_inc += dof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	length=(mem_double+ptr_inc);            ptr_inc += numel;
	lengthn=(mem_double+ptr_inc);           ptr_inc += numel;
	K_diag=(mem_double+ptr_inc);            ptr_inc += dof;
	local_xyz=(mem_double+ptr_inc);         ptr_inc += numel*nsdsq;

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
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;

/* For the XYZI integers */
	                                     ptr_inc = 0;
	bc.fix =(mem_XYZI+ptr_inc);          ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZI+ptr_inc);       ptr_inc += 1;

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

	stress_read_flag = 1;
	check = tsreader( bc, connect, coord, el_matl, force, matl, o1,
		stress, U);
	if(!check) printf( " Problems with tsreader \n");

	check = tslocal_vectors( connect, coord, length, local_xyz );
	if(!check) printf( " Problems with tslocal_vectors \n");

	check = formid( bc, id );
	if(!check) printf( " Problems with formid \n");

#if DATA_ON
	printf( "\n This is the id matrix \n");
	for( i = 0; i < numnp; ++i )
	{
		printf("\n node(%4d)",i);
		for( j = 0; j < ndof; ++j )
		{
			printf(" %4d  ",*(id+ndof*i+j));
		}
	}
#endif

	check = formlm( connect, id, lm, ndof, npel, numel );
	if(!check) printf( " Problems with formlm \n");

#if DATA_ON
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
#endif

	check = diag( idiag, lm, ndof, neqn, npel, numel);
	if(!check) printf( " Problems with diag \n");

#if DATA_ON
	printf( "\n\n This is the idiag matrix \n");
	for( i = 0; i < neqn; ++i )
	{
	    printf( "\ndof %5d   %5d",i,*(idiag+i));
	}
	printf( "\n");
#endif
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
			printf( " LU decomposition\n\n");
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

		sofmA_mass = dof;                       /* mass case 1 */
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
	check = tsKassemble( A, connect, coord, el_matl, force, id, idiag, K_diag,
		lm, length, local_xyz, matl, strain, stress, U);
	if(!check) printf( " Problems with tsKassembler \n");

#if DATA_ON
	printf( "\n\n This is the force matrix \n");
	for( i = 0; i < neqn; ++i )
	{
	    printf( "\ndof %5d   %14.5f",i,*(force+i));
	}
	printf(" \n");
#endif

	if(modal_flag)
	{
	    if( lumped_mass_flag || consistent_mass_store )
	    {

/* For modal analysis, assemble either the diagonal lumped mass matrix or in
   the case of consistent mass, create and store all element mass matrices (if
   there is enough memory).
*/

/* Create mass matrix for modal analysis */

		check = tsMassemble(connect, coord, el_matl, id, length, local_xyz,
			mass, matl);
		if(!check) printf( " Problems with tsMassemble \n");

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

	if(static_flag)
	{
	    if(LU_decomp_flag)
	    {

/* Using LU decomposition to solve the system */

		check = solve(A,force,idiag,neqn);
		if(!check) printf( " Problems with solve \n");
#if DATA_ON
		printf( "\n This is the solution to the problem \n");
#endif
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
		check = tsConjGrad( A, bc, connect, coord, el_matl, force, K_diag,
			length, local_xyz, matl, U);
		if(!check) printf( " Problems with tsConjGrad \n");
	    }

#if DATA_ON
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) > -1 )
		{
			printf("\n node %3d x   %14.6f ",i,*(U+ndof*i));
		}
		if( *(id+ndof*i+1) > -1 )
		{
			printf("\n node %3d y   %14.6f ",i,*(U+ndof*i+1));
		}
		if( *(id+ndof*i+2) > -1 )
		{
			printf("\n node %3d z   %14.6f ",i,*(U+ndof*i+2));
		}
	    }
	    printf(" \n");
#endif

/* Calculate the reaction forces */
	    analysis_flag = 2;
	    memset(force,0,dof*sof);
#if DATA_ON
	    printf( "\n\n These are the axial displacements and forces \n");
#endif
	    check = tsKassemble( A, connect, coord, el_matl, force, id, idiag, K_diag,
		lm, length, local_xyz, matl, strain, stress, U);
	    if(!check) printf( " Problems with tsKassembler \n");
#if DATA_ON
	    printf( "\n\n These are the reaction forces \n");
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) < 0 )
		{
			printf("\n node %3d x   %14.6e ",i,*(force+ndof*i));
		}
		if( *(id+ndof*i+1) < 0 )
		{
			printf("\n node %3d y   %14.6e ",i,*(force+ndof*i+1));
		}
		if( *(id+ndof*i+2) < 0 )
		{
			printf("\n node %3d z   %14.6e ",i,*(force+ndof*i+2));
		}
	    }

	    printf( "\n\n              These are the updated coordinates \n");
	    printf( "\n           x            y             z \n");

	    for( i = 0; i < numnp; ++i )
	    {
		fpointx = *(coord+nsd*i) + *(U+ndof*i);
		fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1);
		fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
		printf("\n node %3d %14.9f %14.9f %14.9f",i,fpointx,fpointy,fpointz);
	    }
	    printf(" \n");
#endif
	    check = tswriter ( bc, connect, coord, el_matl, force, id, matl,
		name, strain, stress, U);
	    if(!check) printf( " Problems with tswriter \n");
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

	    check = tsLanczos(A, bc, connect, coord, eigen, el_matl, id,
		idiag, K_diag, length, local_xyz, mass, matl, ritz, num_eigen);
	    if(!check) printf( " Problems with tsLanczos \n");

/* Write out the eigenvalues to a file */

	    check = eval_data_print( eigen, name, nmode);
	    if(!check) printf( " Problems with eval_data_print \n");

/* Write out the eigenmodes to a (name).mod-x.ots file */

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

/* Re-initialize the stress, strain */

		memset(stress,0,sofmSDIM*sizeof(SDIM));
		memset(strain,0,sofmSDIM*sizeof(SDIM));

/* Calculate the stresses */
		analysis_flag = 2;

		check = tsKassemble( A, connect, coord, el_matl, vector_dum, id,
		    idiag, K_diag, lm, length, local_xyz, matl, strain, stress, U);
		if(!check) printf( " Problems with tsKassemble \n");

		check = tswriter ( bc, connect, coord, el_matl, force, id, matl,
		    name_mode, strain, stress, U);
		if(!check) printf( " Problems with tswriter \n");

		++counter;
	    }
	}
	
/* Calculating the value of the Lengths */

	check = tsLength( connect, coord, lengthn);
	if(!check) printf( " Problems with tsLength \n");

/*
	printf("\nThis is the Length\n");
	for( i = 0; i < numel; ++i )
	{
		printf("%4i %12.4e\n",i, *(lengthn + i));
	}
*/

	timec = clock();
	printf("\n\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(matl);
	free(mem_double);
	free(mem_int);
	if(modal_flag) free(mem_XYZI);
	free(A);
}

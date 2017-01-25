/*
    This program performs finite element analysis
    by reading in the data, doing assembling, and
    then solving the linear system for a brick
    element.

	        Last Update 11/4/09

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
#include "brconst.h"
#include "brstruct.h"

int brConnectSurfwriter ( int *, int *, char *);

int brVolume( int *, double *, double *);

int brwriter ( BOUND , int *, double *, int *, double *, int *, MATL *,
	char *, STRAIN *, SDIM *, STRESS *, SDIM *, double *);

int eval_data_print( EIGEN *, char *, int );

int brLanczos(double *, BOUND , int *, double *, EIGEN *, int *, int *, int *,
	double *, double *, MATL *, double *, int );

int brConjGrad(double *, BOUND , int *, double *, int *, double *, double *, MATL *,
	double *);

int solve(double *,double *,int *, int );

int decomp(double *,int *,int );

int brMassemble(int *, double *, int *, int *, double *, MATL *);

int brKassemble(double *, int *, int *, double *, int *, int *, double *,
	int *, int *, double *, int *, MATL *, double *, STRAIN *, SDIM *,
	STRESS *, SDIM *, double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int formid( BOUND, int *);

int brreader( BOUND , int *, double *, int *, double *, MATL *, char *, FILE *,
	STRESS *, SDIM *, double *);

int Memory( double **, int, int **, int, MATL **, int , XYZI **, int,
	SDIM **, int, STRAIN **, STRESS **, int );

int brshl( double, double *, double * );

int brshl_node2(double * );

int analysis_flag, dof, sdof, modal_flag, neqn, nmat, nmode, numel, numnp, sof;
int static_flag, consistent_mass_flag, consistent_mass_store, eigen_print_flag,
	lumped_mass_flag, stress_read_flag, element_stress_read_flag,
	element_stress_print_flag, gauss_stress_flag;

int LU_decomp_flag, numel_K, numel_P, numel_surf, numnp_LUD_max;
int iteration_max, iteration_const, iteration;
double tolerance;

double shg[sosh], shg_node[sosh], shl[sosh],shl_node[sosh], shl_node2[sosh_node2],
	*Vol0, w[num_int];

int main(int argc, char** argv)
{
	int i, j;
	int *id, *lm, *idiag, check, name_length, counter, MemoryCounter;
	XYZI *mem_XYZI;
	int *mem_int, sofmA, sofmA_K, sofmA_mass, sofmi, sofmf, sofmSTRESS,
		sofmXYZI, sofmSDIM, ptr_inc;
	MATL *matl;
	double *mem_double;
	double fpointx, fpointy, fpointz;
	int *connect, *connect_surf, *el_matl, *el_matl_surf, dum;
	double *coord, *force, *mass, *U, *Voln, *A,
		*node_counter, *vector_dum;
	double *K_diag, *ritz;
	EIGEN *eigen;
	int num_eigen;
	char name[30], buf[ BUFSIZ ];
	char name_mode[30], *ccheck;
	FILE *o1, *o2;
	BOUND bc;
	STRESS *stress;
	STRAIN *strain;
	SDIM *stress_node, *strain_node, *mem_SDIM;
	double g, fdum;
	long timec;
	long timef;
	int  mem_case, mem_case_mass;
	double RAM_max, RAM_usable, RAM_needed, MEGS;

	sof = sizeof(double);

/* Create local shape funcions at gauss points */

	g = 2.0/sq3;
	check = brshl( g, shl, w );
	if(!check) printf( " Problems with brshl \n");

/* Create local shape funcions at nodal points */

	g = 2.0;
	check = brshl( g, shl_node, w );
	if(!check) printf( " Problems with brshl \n");

/* Create local streamlined shape funcion matrix at nodal points */

	check = brshl_node2(shl_node2);
	if(!check) printf( " Problems with brshl_node2 \n");

	memset(name,0,30*sizeof(char));
	
	printf("What is the name of the file containing the \n");
	printf("brick structural data? (example: rubber)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural data */
/*   o2 contains input parameters */

	o1 = fopen( name,"r" );
	o2 = fopen( "brinput","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file brinput\n");
		tolerance = 1.e-13;
		iteration_max = 2000;
		RAM_max = 160.0;
		element_stress_read_flag = 0;
		element_stress_print_flag = 0;
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
		fscanf( o2, "%d\n ",&element_stress_read_flag);
		fgets( buf, BUFSIZ, o2 );
		fscanf( o2, "%d\n ",&element_stress_print_flag);
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

	if(nmode)
	{

/* The criteria for over-calculating the number of desired eigenvalues is
   taken from "The Finite Element Method" by Thomas Hughes, page 578.  It
   is actually given for subspace iteration, but I find it works well
   for the Lanczos Method.  I have also slightly modified
   one of the factors from 2.0 to 2.2.
*/
		num_eigen = (int)(2.2*nmode);
		num_eigen = MIN(nmode + 8, num_eigen);
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
	sofmf=sdof + 3*dof + 2*numel + numnp + dof;
	if(modal_flag)
	{
		sofmf = sdof + 3*dof + 2*numel + numnp + dof + num_eigen*dof;
	}
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi= 2*numel*npel + 2*dof + numel*npel*ndof + 2*numel + numnp+1 + 1;
	MemoryCounter += sofmi*sizeof(int);
	printf( "\n Memory requrement for integers is %15d bytes\n",MemoryCounter);

/* For the XYZI integers */
	sofmXYZI=numnp+1+1;
	MemoryCounter += sofmXYZI*sizeof(XYZI);
	printf( "\n Memory requrement for XYZI integers is %15d bytes\n",MemoryCounter);

/* For the SDIM doubles */
	sofmSDIM = 2*numnp;
	MemoryCounter += sofmSDIM*sizeof(SDIM);
	printf( "\n Memory requrement for SDIM doubles is %15d bytes\n",MemoryCounter);

/* For the STRESS */
	sofmSTRESS=numel;
	MemoryCounter += sofmSTRESS*sizeof(STRESS) + sofmSTRESS*sizeof(STRAIN);
	printf( "\n Memory requrement for STRESS doubles is %15d bytes\n",MemoryCounter);

	check = Memory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZI, sofmXYZI, &mem_SDIM, sofmSDIM, &strain, &stress,
		sofmSTRESS );
	if(!check) printf( " Problems with Memory \n");

/* For the doubles */
	                                        ptr_inc=0; 
	coord=(mem_double+ptr_inc);             ptr_inc += sdof;
	vector_dum=(mem_double+ptr_inc);        ptr_inc += dof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	Voln=(mem_double+ptr_inc);              ptr_inc += numel; 
	Vol0=(mem_double+ptr_inc);              ptr_inc += numel; 
	node_counter=(mem_double+ptr_inc);      ptr_inc += numnp;
	K_diag=(mem_double+ptr_inc);            ptr_inc += dof;

/* If modal analysis is desired, allocate for ritz vectors */

	if(modal_flag)
	{
		ritz=(mem_double+ptr_inc);      ptr_inc += num_eigen*dof; 
	}

/* For the integers */
	                                        ptr_inc = 0; 
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npel; 
	connect_surf=(mem_int+ptr_inc);         ptr_inc += numel*npel;
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
	idiag=(mem_int+ptr_inc);                ptr_inc += dof;
	lm=(mem_int+ptr_inc);                   ptr_inc += numel*npel*ndof;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	el_matl_surf=(mem_int+ptr_inc);         ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;

/* For the XYZI integers */
	                                   ptr_inc = 0; 
	bc.fix =(mem_XYZI+ptr_inc);        ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZI+ptr_inc);     ptr_inc += 1;

/* For the SDIM doubles */
	                                        ptr_inc = 0;
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp;
	strain_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp;

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
	check = brreader( bc, connect, coord, el_matl, force, matl, name, o1,
		stress, stress_node, U);
	if(!check) printf( " Problems with brreader \n");

	printf(" \n\n");

	check = formid( bc, id );
	if(!check) printf( " Problems with formid \n");
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
	check = brKassemble(A, connect, connect_surf, coord, el_matl, el_matl_surf,
		force, id, idiag, K_diag, lm, matl, node_counter, strain,
		strain_node, stress, stress_node, U);
	if(!check) printf( " Problems with brKassemble \n");
/*
	printf( "\n\n This is the force matrix \n");
	for( i = 0; i < neqn; ++i )
	{
	    printf( "\ndof %5d   %14.5f",i,*(force+i));
	}
	printf(" \n");
*/
	if(modal_flag)
	{
	    if( lumped_mass_flag || consistent_mass_store )
	    {

/* For modal analysis, assemble either the diagonal lumped mass matrix or in
   the case of consistent mass, create and store all element mass matrices (if
   there is enough memory).
*/
		check = brMassemble(connect, coord, el_matl, id, mass, matl);
		if(!check) printf( " Problems with brMassemble \n");

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
		check = brConjGrad( A, bc, connect, coord, el_matl, force, K_diag,
			matl, U);
		if(!check) printf( " Problems with brConjGrad \n");
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
	    }
	    printf(" \n");
*/

/* Calculate the reaction forces */
	    analysis_flag = 2;
	    memset(force,0,dof*sof);
	    check = brKassemble(A, connect, connect_surf, coord, el_matl, el_matl_surf,
		force, id, idiag, K_diag, lm, matl, node_counter, strain, strain_node,
		stress, stress_node, U);
	    if(!check) printf( " Problems with brKassembler \n");
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
*/
	    check = brwriter ( bc, connect, coord, el_matl, force, id, matl,
		name, strain, strain_node, stress, stress_node, U);
	    if(!check) printf( " Problems with brwriter \n");
	}

/* If modal analysis desired, calculate the eigenmodes  */

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

/* nmode cannot exceed num_eigen */
	    nmode = MIN(num_eigen, nmode);

	    printf("\n The number of eigenvalues is: %5d \n", num_eigen);

	    if(num_eigen < 2)
	    {
		printf("\n There is only one DOF \n");
		printf("\n so no modal analysis is performed.\n");
		exit(1);
	    }

/* Use Lanczos method for determining eigenvalues */

	    check = brLanczos(A, bc, connect, coord, eigen, el_matl, id, idiag, K_diag,
		mass, matl, ritz, num_eigen);
	    if(!check) printf( " Problems with brLanczos \n");

/* Write out the eigenvalues to a file */

	    check = eval_data_print( eigen, name, nmode);
	    if(!check) printf( " Problems with eval_data_print \n");

/* Write out the eigenmodes to a (name).mod-x.obr file */

	    for( j = 0; j < nmode; ++j )
	    {
		for( i = 0; i < dof; ++i )
		{
		    if( *(id + i) > -1 )
		    {
			/* *(U + i) = *(ritz + num_eigen*(*(id + i))+4);*/
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

/* Re-initialize the stress, strain, stress_node, stain_node */

		memset(stress,0,sofmSTRESS*sizeof(STRESS));
		memset(strain,0,sofmSTRESS*sizeof(STRAIN));
		memset(mem_SDIM,0,sofmSDIM*sizeof(SDIM));

/* Calculate the stresses */
		analysis_flag = 2;

		check = brKassemble(A, connect, connect_surf, coord, el_matl,
		    el_matl_surf, vector_dum, id, idiag, K_diag, lm, matl,
		    node_counter, strain, strain_node, stress, stress_node,
		    U);
		if(!check) printf( " Problems with brKassembler \n");

		check = brwriter( bc, connect, coord, el_matl, force, id, matl,
		    name_mode, strain, strain_node, stress, stress_node, U);
		if(!check) printf( " Problems with brwriter \n");

		++counter;
	    }
	}

/* Calculating the value of the Volumes */

	check = brVolume( connect, coord, Voln);
	if(!check) printf( " Problems with brVolume \n");

/*
	printf("\nThis is the Volume\n");
	for( i = 0; i < numel; ++i )
	{
		printf("%4i %12.4e\n",i, *(Voln + i));
	}
*/

	check = brConnectSurfwriter( connect_surf, el_matl_surf, name);
	if(!check) printf( " Problems with brConnectSurfwriter \n");

	timec = clock();
	printf("\n\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(mem_SDIM);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZI);
	if(modal_flag) free(eigen);
	free(A);
}

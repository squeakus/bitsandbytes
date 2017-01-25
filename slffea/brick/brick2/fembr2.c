/*
    This program performs finite element analysis
    by reading in the data, doing assembling, and
    then solving the linear system for a thermal
    brick element.

	        Updated 12/4/06

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
#include <time.h>
#include "../brick/brconst.h"
#include "br2struct.h"

#define SMALL      1.e-20

int brConnectSurfwriter ( int *, int *, char *);

int brArea( int *, double *, double * );

int brVolume( int *, double *, double *);

int br2writer ( BOUND, int *, int *, double *, int *, int *, double *, double *,
	double *, int *, MATL *, char *, double *, STRAIN *, SDIM *,
	STRESS *, SDIM *, double *, double *, double *);

int br2ConjGrad(double *, BOUND, int *, double *, int *, double *, double *, MATL *,
	double *);

int solve(double *,double *,int *, int );

int decomp(double *,int *,int );

int br2Kassemble(double *, int *, int *, double *, int *, int *, double *, int *,
	int *, double *, int *, MATL *, double *, STRAIN *, SDIM *, STRESS *,
	SDIM *, double *, double *);

int brTConjGrad(double *, BOUND, int *, int *, double *, int *, int *, double *,
	MATL *, double *, double *);

int brCassemble(double *, int *, int *, double *, int *, int *, double *, double *,
	int *, int *, int *, int *, MATL *, double *, double *, double *, double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int brformTid( BOUND, int *);

int br2reader( BOUND , int *, int *, double *, int *, int *, double *, double *,
	double *, MATL *, char *name, FILE *, double *, STRESS *, SDIM *, double *,
	double *, double *);

int Memory( double **, int, int **, int, MATL **, int , XYZI **, int,
	SDIM **, int, STRAIN **, STRESS **, int );

int brshl_film( double , double *);

int brshl( double, double *, double * );

int brshl_node2(double * );

int analysis_flag, temp_analysis_flag, dof, sdof, Tdof, modal_flag, neqn, Tneqn, nmat, nmode,
	numel, numel_film, numnp, sof;
int stress_read_flag, element_stress_read_flag, element_stress_print_flag,
	gauss_stress_flag, disp_analysis, thermal_analysis;

int LU_decomp_flag, TLU_decomp_flag, TBLU_decomp_flag, numel_K, numel_P, Tnumel_K,
	Tnumel_P, TBnumel_K, numel_surf, numnp_LUD_max, Tnumnp_LUD_max;
int iteration_max, iteration_const, iteration;
double tolerance;
	
double shg[sosh], shg_node[sosh], shl[sosh], shl_film[sosh_film], shl_node[sosh],
	shl_node2[sosh_node2], *Vol0, w[num_int];

int main(int argc, char** argv)
{
	int i, j;
	int *id, *lm, *idiag, *Tid, *Tlm, *TBlm, *Tidiag, check, counter, MemoryCounter;
	XYZI *mem_XYZI;
	int *mem_int, sofmA, sofmi, sofmSTRESS, sofmf, sofmXYZI, sofmSDIM, ptr_inc;
	MATL *matl;
	double *mem_double;
	double fpointx, fpointy, fpointz;
	int *connect, *connect_surf, *connect_film, *el_matl, *el_matl_film,
		*el_matl_surf, dum;
	double *coord, *force, *heat_el, *heat_node, *T, *TB, *U, *Voln, *Area, *A,
		*Q, *node_counter;
	double *K_diag, *TK_diag;
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
	int  mem_case;
	double RAM_max, RAM_usable, RAM_needed, MEGS;

	sof = sizeof(double);

/* Create local shape funcions at gauss points */

	memset(shl,0,sosh*sof);
	memset(shl_film,0,sosh_film*sof);
	memset(shl_node,0,sosh*sof);

	g = 2.0/sq3;
	check = brshl( g, shl, w );
	if(!check) printf( " Problems with brshl \n");

	check = brshl_film( g, shl_film );
	if(!check) printf( " Problems with brshl_film \n");

/* Create local shape funcions at nodal points */

	g = 2.0;
	check = brshl( g, shl_node, w );
	if(!check) printf( " Problems with brshl \n");

/* Create local streamlined shape funcion matrix at nodal points */

	check = brshl_node2(shl_node2);
	if(!check) printf( " Problems with brshl_node2 \n");

	memset(name,0,30*sizeof(char));
	
	printf("What is the name of the file containing the \n");
	printf("brick structural data? (example: fins4)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural data */
/*   o2 contains input parameters */

	o1 = fopen( name,"r" );
	o2 = fopen( "br2input","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file br2input\n");
		tolerance = 1.e-13;
		iteration_max = 2000;
		RAM_max = 160.0;
		element_stress_read_flag = 0;
		element_stress_print_flag = 0;
		gauss_stress_flag = 0;
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
	}

	fgets( buf, BUFSIZ, o1 );
	fscanf( o1, "%d %d %d %d %d %d\n ", &numel, &numnp, &nmat, &numel_film,
		&disp_analysis, &thermal_analysis);
	Tdof=numnp*Tndof;
	dof=numnp*ndof;
	sdof=numnp*nsd;

	numnp_LUD_max = 750;
	Tnumnp_LUD_max = 3*numnp_LUD_max;

/* Assuming Conjugate gradient method is used, determine how much RAM is needed.
   This determines the largest problem that can be run on this machine.
   If problem small enough that LU decoomposition is used, then calculation below
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

/* Because neqlsq = 576 and Tneqel = 64, we can store 9X as many temperature
   element stiffness matrices as that of displacement stiffnesses.
*/

	Tnumel_K = numel;
	TBnumel_K = numel_film;
	dum = 9*numel_K;
	Tnumel_P = dum - numel;

	if( Tnumel_P > 0)
	{
/* Enough RAM to store all element temperature stiffness matrices for
   conjugate gradient*/

		Tnumel_P = 0;

		dum = 9*numel_K - numel - (int)(.5*((double)numel_film));
		if( dum < 0  )
		{
/* Cannot store film stiffness elements */
			TBnumel_K = 0;
		}
	}
	else
	{
		Tnumel_P = numel - (((int)RAM_usable)/((double)Tneqlsq*sof));
		Tnumel_K = numel - Tnumel_P;

/* Cannot store film stiffness elements */

		TBnumel_K = 0;
	}

	LU_decomp_flag = 1;
	if(numnp > numnp_LUD_max) LU_decomp_flag = 0;
	TLU_decomp_flag = 1;
	TBLU_decomp_flag = 1;
	if(numnp > Tnumnp_LUD_max)
	{
		TLU_decomp_flag = 0;
		TBLU_decomp_flag = 0;
	}

#if 0
	LU_decomp_flag = 0;
	TLU_decomp_flag = 0;
#endif

/*   Begin allocation of meomory */

	MemoryCounter = 0;

/* For the doubles */
	sofmf=sdof + dof + numel + 4*Tdof + dof + numel_film + 2*numel +
		numnp + dof + Tdof;
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi= 2*numel*npel + numel_film*npel_film + 2*dof + numel*npel*ndof +
		2*Tdof + numel*npel*Tndof + numel_film*npel_film*Tndof + numel +
		numel_film + numel + numnp+1 + numel+1 + 4*numnp + 4 + 6;
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

/* For the STRESS doubles */
	sofmSTRESS = numel;
	MemoryCounter += sofmSTRESS*sizeof(STRESS) + sofmSTRESS*sizeof(STRAIN); 
	printf( "\n Memory requrement for STRESS doubles is %15d bytes\n",MemoryCounter);

	check = Memory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZI, sofmXYZI, &mem_SDIM, sofmSDIM, &strain, &stress, sofmSTRESS );
	if(!check) printf( " Problems with Memory \n");

/* For the doubles */
	                                        ptr_inc=0; 
	coord=(mem_double+ptr_inc);             ptr_inc += sdof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	heat_el=(mem_double+ptr_inc);           ptr_inc += numel;
	heat_node=(mem_double+ptr_inc);         ptr_inc += Tdof;
	Q=(mem_double+ptr_inc);	                ptr_inc += Tdof; 
	T=(mem_double+ptr_inc);                 ptr_inc += Tdof;
	TB=(mem_double+ptr_inc);                ptr_inc += Tdof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	Area=(mem_double+ptr_inc);              ptr_inc += numel_film; 
	Voln=(mem_double+ptr_inc);              ptr_inc += numel; 
	Vol0=(mem_double+ptr_inc);	        ptr_inc += numel; 
	node_counter=(mem_double+ptr_inc);      ptr_inc += numnp;
	K_diag=(mem_double+ptr_inc);            ptr_inc += dof;
	TK_diag=(mem_double+ptr_inc);           ptr_inc += Tdof;

/* For the integers */
	                                        ptr_inc = 0; 
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npel; 
	connect_surf=(mem_int+ptr_inc);         ptr_inc += numel*npel; 
	connect_film=(mem_int+ptr_inc);         ptr_inc += numel_film*npel_film; 
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
	idiag=(mem_int+ptr_inc);                ptr_inc += dof;
	lm=(mem_int+ptr_inc);                   ptr_inc += numel*npel*ndof;
	Tid=(mem_int+ptr_inc);                  ptr_inc += Tdof;
	Tidiag=(mem_int+ptr_inc);               ptr_inc += Tdof;
	Tlm=(mem_int+ptr_inc);                  ptr_inc += numel*npel*Tndof;
	TBlm=(mem_int+ptr_inc);                 ptr_inc += numel_film*npel_film*Tndof;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	el_matl_film=(mem_int+ptr_inc);         ptr_inc += numel_film;
	el_matl_surf=(mem_int+ptr_inc);         ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.heat_el =(mem_int+ptr_inc);          ptr_inc += numel+1;
	bc.heat_node =(mem_int+ptr_inc);        ptr_inc += numnp+1;
	bc.Q =(mem_int+ptr_inc);                ptr_inc += numnp+1;
	bc.T =(mem_int+ptr_inc);                ptr_inc += numnp+1;
	bc.TB =(mem_int+ptr_inc);               ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;
	bc.num_heat_el=(mem_int+ptr_inc);       ptr_inc += 1;
	bc.num_heat_node=(mem_int+ptr_inc);     ptr_inc += 1;
	bc.num_Q=(mem_int+ptr_inc);             ptr_inc += 1;
	bc.num_T=(mem_int+ptr_inc);             ptr_inc += 1;
	bc.num_TB=(mem_int+ptr_inc);            ptr_inc += 1;

/* For the XYZI integers */
	                                   ptr_inc = 0;
	bc.fix =(mem_XYZI+ptr_inc);        ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZI+ptr_inc);     ptr_inc += 1;

/* For the SDIM doubles */
	                                        ptr_inc = 0; 
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp; 
	strain_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp;

	timec = clock();
	timef = 0;

	stress_read_flag = 1;

	check = br2reader( bc, connect, connect_film, coord, el_matl, el_matl_film,
		force, heat_el, heat_node, matl, name, o1, Q, stress, stress_node, T,
		TB, U);
	if(!check) printf( " Problems with br2reader \n");

	printf(" \n\n");

/* form Tid, Tlm, Tidiag matrix for Temperature calculation */

	check = brformTid( bc, Tid );
	if(!check) printf( " Problems with brformTid \n");
/*
	printf( "\n This is the Tid matrix \n");
	for( i = 0; i < numnp; ++i )
	{
		printf("\n node(%4d)",i); 
		for( j = 0; j < Tndof; ++j )
		{
			printf(" %4d  ",*(Tid+Tndof*i+j));
		}
	}
*/
	check = formlm( connect, Tid, Tlm, Tndof, npel, numel );
	if(!check) printf( " Problems with formlm \n");
/*
	printf( "\n\n This is the Tlm matrix \n");
	for( i = 0; i < numel; ++i )
	{
	    printf("\n element(%4d)",i);
	    for( j = 0; j < Tneqel; ++j )
	    {
		printf( "%5d   ",*(Tlm+Tneqel*i+j));
	    }
	}
	printf( "\n");
*/

	check = formlm( connect_film, Tid, TBlm, Tndof, npel_film, numel_film );
	if(!check) printf( " Problems with formlm \n");
/*
	printf( "\n\n This is the TBlm matrix \n");
	for( i = 0; i < numel_film; ++i )
	{
	    printf("\n surface element(%4d)",i);
	    for( j = 0; j < TBneqel; ++j )
	    {
		printf( "%5d   ",*(TBlm+TBneqel*i+j));
	    }
	}
	printf( "\n");
*/

	check = diag( Tidiag, Tlm, Tndof, Tneqn, npel, numel);
	if(!check) printf( " Problems with diag \n");

/*
	printf( "\n\n This is the Tidiag matrix \n");
	for( i = 0; i < Tneqn; ++i )
	{
	    printf( "\nTdof %5d   %5d",i,*(Tidiag+i));
	}
	printf( "\n");
*/


/* form id, lm, idiag matrix for Displacement calculation. */

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


/*
     Allocate meomory for A.  This A is used for both the temperature and displacement
     calculation, so we need to find which calculation will require more memory.  There
     are 3 possibilities:

     1) LU decoomposition for both, in which case allocate for the displacement calculation
	which is bigger.
	  1a)Skyline storage of global conductivity matrix
	  1b)Skyline storage of global stiffness mmatrix

     2) LU decoomposition for temperature and conjugate gradient for displacement.  For this
	case, we need to carefully examine which needs more memory.
	  2a)Skyline storage of global conductivity matrix
	  2b)storage of numel_K element stiffness matrices

     3) Conjugate gradient for both.
	  3a)storage of Tnumel_K element conductivity matrices 
	  3b)storage of numel_K element stiffness matrices

*/
	sofmA = numel_K*neqlsq;                  /* case 3 */
	dum = Tnumel_K*Tneqlsq + TBnumel_K*TBneqlsq; 
	if( sofmA < dum ) sofmA = dum; 
	mem_case = 3;

	if(TLU_decomp_flag)
	{
		dum = *(Tidiag+Tneqn-1)+1;       /* case 2 */
		if( sofmA < dum ) sofmA = dum; 
		mem_case = 2;
	}
	if(LU_decomp_flag)
	{
		sofmA = *(idiag+neqn-1)+1;       /* case 1 */
		mem_case = 1;
	}

	if( sofmA*sof > (int)RAM_usable )
	{

/* Even if LU decoomposition flags are on because there are only a few nodes, there
   is a possibility that there is not enough memory because of poor node numbering.
   If this is the case, then we have to use the conjugate gradient method.
 */

		sofmA = numel_K*neqlsq;
		dum = Tnumel_K*Tneqlsq + TBnumel_K*TBneqlsq; 
		if( sofmA < dum ) sofmA = dum; 

		LU_decomp_flag = 0;
		TLU_decomp_flag = 0;
		TBLU_decomp_flag = 0;
		mem_case = 3;
	}

	printf( "\n We are in case %3d\n\n", mem_case);
	switch (mem_case) {
		case 1:
			printf( " LU decoomposition for both\n\n");
		break;
		case 2:
			printf( " LU decoomposition for temperature\n");
			printf( " Conjugate gradient for displacement\n\n");
		break;
		case 3:
			printf( " Conjugate gradient for both\n\n");
		break;
	}


	MemoryCounter += sofmA*sizeof(double);
	printf( "\n Memory requrement for disp calc. with %15d doubles is %15d bytes\n",
		sofmA, MemoryCounter);
	MEGS = ((double)(MemoryCounter))/MB;
	printf( "\n Which is %16.4e MB\n", MEGS);
	printf( "\n This is numel, numel_K, numel_P %5d %5d %5d\n", numel, numel_K, numel_P);
	printf( "\n This is numel, Tnumel_K, Tnumel_P %5d %5d %5d\n", numel, Tnumel_K, Tnumel_P);

/* Note that A for the stiffness matrix will not be needed until later */

	if(sofmA < 1) sofmA = 1;
	A=(double *)calloc(sofmA,sizeof(double));
	if(!A )
	{
		printf( "failed to allocate memory for A double for displacement calc.\n ");
		exit(1);
	}



	if(thermal_analysis > 0)
	{	
	   temp_analysis_flag = 1;
	   memset(A,0,sofmA*sof);
	   memset(TK_diag,0,Tdof*sof);
	   check = brCassemble(A, connect, connect_film, coord, el_matl, el_matl_film,
		heat_el, heat_node, Tid, Tidiag, Tlm, TBlm, matl, Q, T, TB, TK_diag);
	   if(!check) printf( " Problems with brCassemble \n");

	   if(TLU_decomp_flag)
	   {
/*
		printf( "\n\n This is the heat Q matrix \n");
		for( i = 0; i < Tneqn; ++i )
		{
			printf( "\Tndof %5d   %14.5f",i,*(Q+i));
		}
		printf(" \n");
*/
/* Perform LU Crout decompostion on the system */

		check = decomp(A,Tidiag,Tneqn);
		if(!check) printf( " Problems with decomp \n");

		check = solve(A,Q,Tidiag,Tneqn);
		if(!check) printf( " Problems with solve \n");

/* printf( "\n This is the solution to the problem \n");*/

		for( i = 0; i < Tdof; ++i )
		{
			if( *(Tid + i) > -1 )
			{
				*(T + i) = *(Q + *(Tid + i));
			}
		}
	   }

/* Using Conjugate gradient method to find temperature distribution */

	   if(!TLU_decomp_flag)
	   {
		check = brTConjGrad( A, bc, connect, connect_film, coord, el_matl,
			el_matl_film, Q, matl, T, TK_diag);
		if(!check) printf( " Problems with brTConjGrad \n");
	   }

/*
	   for( i = 0; i < Tdof; ++i )
	   {
		if( *(Tid+Tndof*i) > -1 )
		{
			printf("\n node %3d T   %14.6e ",i,*(T+Tndof*i));
		}
	   }
	   printf(" \n");
*/

/* Calculate the reaction heat */
	   temp_analysis_flag = 2;
	   memset(Q,0,Tdof*sof);
	   check = brCassemble(A, connect, connect_film, coord, el_matl, el_matl_film,
		heat_el, heat_node, Tid, Tidiag, Tlm, TBlm, matl, Q, T, TB, TK_diag);
	   if(!check) printf( " Problems with brCassembler \n");
/*
	   printf( "\n\n These are the reaction heat Q\n");
	   for( i = 0; i < numnp; ++i )
	   {
		if( *(Tid+Tndof*i) < 0 )
		{
			printf("\n node %3d x   %14.6f ",i,*(Q+Tndof*i));
		}
	   }
*/
	}


	if(disp_analysis > 0)
	{	
	   analysis_flag = 1;
	   memset(A,0,sofmA*sof);
	   memset(K_diag,0,dof*sof);
	   check = br2Kassemble(A, connect, connect_surf, coord, el_matl, el_matl_surf,
		force, id, idiag, K_diag, lm, matl, node_counter, strain,
		strain_node, stress, stress_node, T, U);
	   if(!check) printf( " Problems with br2Kassemble \n");

	   if(LU_decomp_flag)
	   {
/* Perform LU Crout decompostion on the system */

		check = decomp(A,idiag,neqn);
		if(!check) printf( " Problems with decomp \n");

/* Solve the system */

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

/* Using Conjugate gradient method to find displacements */

	   if(!LU_decomp_flag)
	   {
		check = br2ConjGrad( A, bc, connect, coord, el_matl, force, K_diag,
			matl, U);
		if(!check) printf( " Problems with br2ConjGrad \n");
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
	   printf(" \n");

/* Calculate the reaction forces */

	   analysis_flag = 2;
	   memset(force,0,dof*sof);
	   check = br2Kassemble(A, connect, connect_surf, coord, el_matl, el_matl_surf,
		force, id, idiag, K_diag, lm, matl, node_counter, strain,
		strain_node, stress, stress_node, T, U);
	   if(!check) printf( " Problems with br2Kassemble \n");
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

/* Calculating the value of the convection film surface Areas */

	if(thermal_analysis == 1)
	{
		check = brArea( connect_film, coord, Area );
		if(!check) printf( " Problems with brArea \n");
	}

/*
	printf("\nThis is the Covection Surfaces\n");
	for( i = 0; i < numel_surf; ++i )
	{
		printf("%4i %12.4e\n",i, *(Area + i));
	}
*/

	check = br2writer ( bc, connect, connect_film, coord, el_matl, el_matl_film,
		force, heat_el, heat_node, id, matl, name, Q, strain, strain_node,
		stress, stress_node, T, TB, U);
	if(!check) printf( " Problems with br2writer \n");

	if(disp_analysis)
	{
		check = brConnectSurfwriter( connect_surf, el_matl_surf, name);
		if(!check) printf( " Problems with brConnectSurfwriter \n");
	}

	timec = clock();
	printf("\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(mem_SDIM);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZI);
	free(A);
}

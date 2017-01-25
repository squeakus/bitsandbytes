/*
    This program performs finite element analysis by reading in
    the data, doing assembling, and then solving the linear system
    for a 3 or 4 node doubly curved shell element.  The shell itself may
    be defined by 6 or 8 nodes.  Alternatively, as of April 2005, this
    element also works for shells defined by 4 nodes only and as of
    October 2008, works for shells defined by 3 nodes only.  Note that
    for the 3 or 4 node shells, the element thickness does not mean that
    these elements are of constant thickness.  Rather, each element
    thickness is averaged over all elements which share a node, and
    this is used to calculate the corresponding top nodes for the
    entire mesh.

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
#include "shconst.h"
#include "shstruct.h"

int shVolume( int *, double *, double *);

int shwriter( BOUND , int *, double *, int *, double *, int *, double *,
	MATL *, char *, STRAIN *, SDIM *, STRESS *, SDIM *, double *, double *);

int eval_data_print( EIGEN *, char *, int );

int shLanczos(double *, BOUND , int *, double *, EIGEN *, int *, int *,
	int *, double *, double *, double *, double *, MATL *, double *, int );

int shConjGrad(double *, BOUND , int *, double *, int *, double *, double *, double *,
	double *, MATL *, double *);

int dotX(double *,double *, double *, int );

int solve(double *,double *,int *, int );

int decomp(double *,int *,int );

int shMassemble(int *, double *, int *, int *, double *, double *, double *, MATL *);

int shKassemble(double *, int *, double *, int *, double *, double *, int *, int *,
	double *, int *, double*, double *, MATL *, double *, STRAIN *, SDIM *, STRESS *,
	SDIM *, double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int shformid( BOUND, int *);

int shlocal_vectors( double *, double *, double * );

int shTopCoordinates(int *, double *, int *, double *, MATL *, double *);

int node_normals( int *, double *, double *, double *, int, int);

int shreader( BOUND , int *, double *, int *, double *, MATL *, char *,
	FILE *, STRESS *, SDIM *, double *);

int shMemory( double **, int , int **, int , MATL **, int , XYZPhiI **,
	int , SDIM **, int , STRAIN **, STRESS **, int  );

int shshl( double, SH, double * );

int shshl_triangle( double, SH, double * );

#if 0
int shshl_node2(double * );
#endif

int analysis_flag, dof, sdof, modal_flag, integ_flag, doubly_curved_flag,
	neqn, nmat, nmode, numel, numnp, sof, flag_quad_element;
int static_flag, consistent_mass_flag, consistent_mass_store, eigen_print_flag,
	lumped_mass_flag, stress_read_flag, element_stress_read_flag,
	element_stress_print_flag, gauss_stress_flag;

int LU_decomp_flag, numel_K, numel_P, numnp_LUD_max;
int iteration_max, iteration_const, iteration;
double tolerance;

SH shg, shg_node, shl, shl_node, shl_tri, shl_tri_node;
ROTATE rotate, rotate_node;
double shl_node2[sosh_node2], *Vol0, w[num_int8], w_tri[num_int6];

int main(int argc, char** argv)
{
	int i, j, k;
	int *id, *lm, *idiag, check, name_length, counter, MemoryCounter;
	XYZPhiI *mem_XYZPhiI;
	int *mem_int, sofmA, sofmA_K, sofmA_mass, sofmi, sofmf, sofmSTRESS,
		sofmshf, sofmrotf, sofmXYZPhiI, sofmSDIM, ptr_inc;
	MATL *matl;
	double *mem_double, *mem_sh_double, *mem_rotate_double;
	double fpointx, fpointy, fpointz, node_vec[nsd];
	int *connect, *el_matl, dum;
	double *coord, *force, *mass, *U, *Uz_fib, *Voln, *A,
		*node_counter, *vector_dum, *fiber_vec, *fiber_xyz, *lamina_ref;
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
	double thickness;
	int npell_, neqel_;

	sof = sizeof(double);

/* For the ROTATE doubles */

	sofmrotf=2*sorls + 2*sorlb72 + 2*sorfs36;
	mem_rotate_double=(double *)calloc(sofmrotf,sizeof(double));

	if(!mem_rotate_double )
	{
		printf( "failed to allocate memory for ROTATE doubles\n ");
		exit(1);
	}
	                                                     ptr_inc=0;
	rotate.l_bend=(mem_rotate_double+ptr_inc);           ptr_inc += sorlb72;
	rotate.l_shear=(mem_rotate_double+ptr_inc);          ptr_inc += sorls;
	rotate.f_shear=(mem_rotate_double+ptr_inc);          ptr_inc += sorfs36;

	rotate_node.l_bend=(mem_rotate_double+ptr_inc);      ptr_inc += sorlb72;
	rotate_node.l_shear=(mem_rotate_double+ptr_inc);     ptr_inc += sorls;
	rotate_node.f_shear=(mem_rotate_double+ptr_inc);     ptr_inc += sorfs36;

/* For the SH doubles */
	sofmshf=2*soshlb48 + 2*soshls12 + 2*soshl_zb + 2*soshl_zs +
		2*soshlb27 + 2*soshls9 + 2*soshl_zb + 2*soshl_zs +
		2*soshgb128 + 2*soshgs32 + 2*soshg_zb96 + 2*soshg_zs24;
	mem_sh_double=(double *)calloc(sofmshf,sizeof(double));

	if(!mem_sh_double )
	{
		printf( "failed to allocate memory for SH double\n ");
		exit(1);
	}
	                                                ptr_inc=0;
	shl.bend=(mem_sh_double+ptr_inc);               ptr_inc += soshlb48;
	shl.shear=(mem_sh_double+ptr_inc);              ptr_inc += soshls12;
	shl.bend_z=(mem_sh_double+ptr_inc);             ptr_inc += soshl_zb;
	shl.shear_z=(mem_sh_double+ptr_inc);            ptr_inc += soshl_zs;

	shl_node.bend=(mem_sh_double+ptr_inc);          ptr_inc += soshlb48;
	shl_node.shear=(mem_sh_double+ptr_inc);         ptr_inc += soshls12;
	shl_node.bend_z=(mem_sh_double+ptr_inc);        ptr_inc += soshl_zb;
	shl_node.shear_z=(mem_sh_double+ptr_inc);       ptr_inc += soshl_zs;

	shl_tri.bend=(mem_sh_double+ptr_inc);           ptr_inc += soshlb27;
	shl_tri.shear=(mem_sh_double+ptr_inc);          ptr_inc += soshls9;
	shl_tri.bend_z=(mem_sh_double+ptr_inc);         ptr_inc += soshl_zb;
	shl_tri.shear_z=(mem_sh_double+ptr_inc);        ptr_inc += soshl_zs;

	shl_tri_node.bend=(mem_sh_double+ptr_inc);      ptr_inc += soshlb27;
	shl_tri_node.shear=(mem_sh_double+ptr_inc);     ptr_inc += soshls9;
	shl_tri_node.bend_z=(mem_sh_double+ptr_inc);    ptr_inc += soshl_zb;
	shl_tri_node.shear_z=(mem_sh_double+ptr_inc);   ptr_inc += soshl_zs;

	shg.bend=(mem_sh_double+ptr_inc);               ptr_inc += soshgb128;
	shg.shear=(mem_sh_double+ptr_inc);              ptr_inc += soshgs32;
	shg.bend_z=(mem_sh_double+ptr_inc);             ptr_inc += soshg_zb96;
	shg.shear_z=(mem_sh_double+ptr_inc);            ptr_inc += soshg_zs24;

	shg_node.bend=(mem_sh_double+ptr_inc);          ptr_inc += soshgb128;
	shg_node.shear=(mem_sh_double+ptr_inc);         ptr_inc += soshgs32;
	shg_node.bend_z=(mem_sh_double+ptr_inc);        ptr_inc += soshg_zb96;
	shg_node.shear_z=(mem_sh_double+ptr_inc);       ptr_inc += soshg_zs24;

/* Create local shape funcions at gauss points for quad elements */

	g = 2.0/sq3;
	check = shshl( g, shl, w );
	if(!check) printf( " Problems with shshl \n");

/* Create local shape funcions at nodal points for quad elements */

	g = 2.0;
	check = shshl( g, shl_node, w );
	if(!check) printf( " Problems with shshl \n");

/* Create local shape funcions at gauss points for triangle elements */

	g = 2.0/sq3;
	check = shshl_triangle( g, shl_tri, w_tri );
	if(!check) printf( " Problems with shshl_triangle \n");

/* Create local shape funcions at nodal points for triangle elements */

	g = 2.0;
	check = shshl_triangle( g, shl_tri_node, w_tri );
	if(!check) printf( " Problems with shshl_triangle \n");

/* Create local streamlined shape funcion matrix at nodal points */

#if 0
	check = shshl_node2(shl_node2);
	if(!check) printf( " Problems with shshl_node2 \n");
#endif

	memset(name,0,30*sizeof(char));
	
	printf("What is the name of the file containing the \n");
	printf("shell structural data? (example: bubble4node)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural shell data  */
/*   o2 contains input parameters */

	o1 = fopen( name,"r" );
	o2 = fopen( "shinput","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file shinput\n");
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
	fscanf( o1, "%d %d %d %d %d\n ",&numel,&numnp,&nmat,&nmode,&integ_flag);
	dof=numnp*ndof;
	sdof=numnp*nsd;

	numnp_LUD_max = 450;

/* Assuming Conjugate gradient method is used, determine how much RAM is needed.
   This determines the largest problem that can be run on this machine.
   If problem small enough that LU decomposition is used, then calculation below
   is irrelevant.

   RAM variables given in bytes

*/
	RAM_max *= MB;
	RAM_usable = 0.5*RAM_max;
	RAM_needed = numel*neqlsq400*sof;

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

		numel_P = numel - (((int)RAM_usable)/((double)neqlsq400*sof));
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
	sofmf=2*sdof + 3*dof + numnp + 2*numel + 2*numnp + dof + sdof + numnp*nsdsq +
		numnp*nsd;
	if(modal_flag)
	{
		sofmf=2*sdof + 3*dof + numnp + 2*numel + 2*numnp + dof +
			sdof + numnp*nsdsq + numnp*nsd + num_eigen*dof;
	}
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi= numel*npell4 + 2*dof + numel*npell4*ndof + numel + numnp+1+1;
	MemoryCounter += sofmi*sizeof(int);
	printf( "\n Memory requrement for integers is %15d bytes\n",MemoryCounter);

/* For the XYZPhiI integers */
	sofmXYZPhiI=numnp+1+1;
	MemoryCounter += sofmXYZPhiI*sizeof(XYZPhiI);
	printf( "\n Memory requrement for XYZPhiI integers is %15d bytes\n",MemoryCounter);

/* For the SDIM doubles */
	sofmSDIM = 2*2*numnp;
	MemoryCounter += sofmSDIM*sizeof(SDIM);
	printf( "\n Memory requrement for SDIM doubles is %15d bytes\n",MemoryCounter);

/* For the STRESS */
	sofmSTRESS=numel;
	MemoryCounter += sofmSTRESS*sizeof(STRESS) + sofmSTRESS*sizeof(STRAIN);
	printf( "\n Memory requrement for STRESS doubles is %15d bytes\n",MemoryCounter);

	check = shMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZPhiI, sofmXYZPhiI, &mem_SDIM, sofmSDIM, &strain,
		&stress, sofmSTRESS );
	if(!check) printf( " Problems with shMemory \n");

/* For the doubles */
	                                        ptr_inc=0;
	coord=(mem_double+ptr_inc);             ptr_inc += 2*sdof;
	vector_dum=(mem_double+ptr_inc);        ptr_inc += dof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	Uz_fib=(mem_double+ptr_inc);            ptr_inc += numnp;
	Voln=(mem_double+ptr_inc);              ptr_inc += numel;
	Vol0=(mem_double+ptr_inc);              ptr_inc += numel;
	node_counter=(mem_double+ptr_inc);      ptr_inc += 2*numnp;
	K_diag=(mem_double+ptr_inc);            ptr_inc += dof;
	fiber_vec=(mem_double+ptr_inc);         ptr_inc += sdof;
	fiber_xyz=(mem_double+ptr_inc);         ptr_inc += numnp*nsdsq;
	lamina_ref=(mem_double+ptr_inc);        ptr_inc += numnp*nsd;

/* If modal analysis is desired, allocate for ritz vectors */

	if(modal_flag)
	{
		ritz=(mem_double+ptr_inc);      ptr_inc += num_eigen*dof;
	}

/* For the integers */
	                                        ptr_inc = 0; 
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npell4; 
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
	idiag=(mem_int+ptr_inc);                ptr_inc += dof;
	lm=(mem_int+ptr_inc);                   ptr_inc += numel*npell4*ndof;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;

/* For the XYZPhiI integers */
	                                      ptr_inc = 0; 
	bc.fix =(mem_XYZPhiI+ptr_inc);        ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZPhiI+ptr_inc);     ptr_inc += 1;

/* For the SDIM doubles */
                                                ptr_inc = 0;
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += 2*numnp;
	strain_node=(mem_SDIM+ptr_inc);         ptr_inc += 2*numnp;

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
	check = shreader( bc, connect, coord, el_matl, force, matl, name,
		o1, stress, stress_node, U);
	if(!check) printf( " Problems with shreader \n");

	npell_ = npell4;
	neqel_ = neqel20;
	if(!flag_quad_element)
	{
		npell_ = npell3;
		neqel_ = neqel15;
	}

	if(!doubly_curved_flag)
	{
/*  In the case of a singly curved shell, I use the function node_normals calculate
    vectors in the fiber direction for each node.
*/ 
	    check = node_normals( connect, coord, fiber_vec, node_counter,
		numel, numnp);
	    if(!check) printf( " Problems with fiber_vec \n");

	    shTopCoordinates(connect, coord, el_matl, fiber_vec, matl, node_counter);
	    if(!check) printf( " Problems with shTopCoordinates\n");

	    memset(node_counter,0,numnp*sizeof(double));
	}

	check = shlocal_vectors( coord, lamina_ref, fiber_xyz );
	if(!check) printf( " Problems with shlocal_vectors\n");

	printf(" \n\n");

	check = shformid( bc, id );
	if(!check) printf( " Problems with shformid \n");
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
	check = formlm( connect, id, lm, ndof, npell_, numel );
	if(!check) printf( " Problems with formlm \n");
/*
	printf( "\n\n This is the lm matrix \n");
	for( i = 0; i < numel; ++i )
	{
	    printf("\n element(%4d)",i);
	    for( j = 0; j < neqel_; ++j )
	    {
		printf( "%5d",*(lm+neqel_*i+j));
	    }
	}
	printf( "\n");
*/
	check = diag( idiag, lm, ndof, neqn, npell_, numel);
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
	sofmA = numel_K*neqlsq400;                  /* case 2 */
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

		sofmA = numel_K*neqlsq400;
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
		fdum = RAM_usable - (double)(sof*(sofmA + numel*neqlsq400));
		if(fdum > 0.0)
		{
/* Enough RAM to store all element mass matrices for modal analysis */

			sofmA_mass = numel*neqlsq400;      /* mass case 2 */
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

	check = shKassemble(A, connect, coord, el_matl, fiber_vec, force,
		id, idiag, K_diag, lm, lamina_ref, fiber_xyz, matl, node_counter,
		strain, strain_node, stress, stress_node, U);
	if(!check) printf( " Problems with shKassembler \n");

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
		check = shMassemble(connect, coord, el_matl, id, lamina_ref,
			fiber_xyz, mass, matl);
		if(!check) printf( " Problems with shMassemble \n");

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

/*
	for( i = 0; i < *(idiag+neqn-1)+1; ++i )
	{
		printf( "\n a[%3d] = %12.5e",i,*(A+i));
	}
	for( i = 0; i < neqn; ++i )
	{
		printf( "\n force[%3d] = %12.5e",i,*(force+i));
	}
*/

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
		check = shConjGrad( A, bc, connect, coord, el_matl, force, K_diag,
			lamina_ref, fiber_xyz, matl, U);
		if(!check) printf( " Problems with shConjGrad \n");
	    }
/*
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) > -1 )
		{
			printf("\nnode %2d x     %14.5e ",i,*(U+ndof*i));
		}
		if( *(id+ndof*i+1) > -1 )
		{
			printf("\nnode %2d y     %14.5e ",i,*(U+ndof*i+1));
		}
		if( *(id+ndof*i+2) > -1 )
		{
			printf("\nnode %2d z     %14.5e ",i,*(U+ndof*i+2));
		}
		if( *(id+ndof*i+3) > -1 )
		{
			printf("\nnode %2d phi x %14.5e ",i,*(U+ndof*i+3));
		}
		if( *(id+ndof*i+4) > -1 )
		{
			printf("\nnode %2d phi y %14.5e ",i,*(U+ndof*i+4));
		}
	    }
	    printf(" \n");
*/
/* Calculate the reaction forces */
	    analysis_flag = 2;
	    memset(force,0,dof*sof);

	    check = shKassemble(A, connect, coord, el_matl, fiber_vec, force,
		    id, idiag, K_diag, lm, lamina_ref, fiber_xyz, matl, node_counter,
		    strain, strain_node, stress, stress_node, U);
	    if(!check) printf( " Problems with shKassembler \n");

/* Calcuate the local z fiber displacement on each node */

	    for( i = 0; i < numnp; ++i )
	    {
/* Calcuate the projection of displacement vector to local z fiber vector */

		*(node_vec) = *(coord+nsd*(i+numnp))-*(coord+nsd*i);
		*(node_vec+1) = *(coord+nsd*(i+numnp)+1)-*(coord+nsd*i+1);
		*(node_vec+2) = *(coord+nsd*(i+numnp)+2)-*(coord+nsd*i+2);
		fdum = *(node_vec)*(*(node_vec)) +
			*(node_vec+1)*(*(node_vec+1)) + *(node_vec+2)*(*(node_vec+2));
		fdum = sqrt(fdum);
		*(node_vec) /= fdum;
		*(node_vec+1) /= fdum;
		*(node_vec+2) /= fdum;
		check = dotX( (Uz_fib+i), (node_vec), (U+ndof*i), nsd);
		if(!check) printf( " Problems with dotX \n");
		/* *(Uz_fib+i) *= .5;*/
		/*printf("\n node %3d %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e",
			i,*(Uz_fib+i),*(U+ndof*i),*(U+ndof*i+1),*(U+ndof*i+2),
			*(node_vec), *(node_vec+1), *(node_vec+2),fdum);*/
	    }
/*
	    printf( "\n\n These are the reaction forces and moments\n");
	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) < 0 )
		{
			printf("\n node %3d z       %14.6f ",i,*(force+ndof*i));
		}
		if( *(id+ndof*i+1) < 0 )
		{
			printf("\n node %3d x       %14.6f ",i,*(force+ndof*i+1));
		}
		if( *(id+ndof*i+2) < 0 )
		{
			printf("\n node %3d y       %14.6f ",i,*(force+ndof*i+2));
		}
		if( *(id+ndof*i+3) < 0 )
		{
			printf("\n node %3d phi x   %14.6f ",i,*(force+ndof*i+3));
		}
		if( *(id+ndof*i+4) < 0 )
		{
			printf("\n node %3d phi y   %14.6f ",i,*(force+ndof*i+4));
		}
	    }
*/
/* This method for calculating the updated coordinates is based on equations 6.2.20-6.2.24 of
   "The Finite Element Method" by Thomas Hughes, page 388-389.  (David Benson, in "AMES 232 B
   Course Notes" page 50, does the same as Hughes.)  I have made some modifications,
   specifically, I replace za in equation 6.2.23 with "Uz_fib" which is the projection of the
   displacement at a node in the fiber "e3" direction.

   It should be noted that in the "ANSYS User's Manual", page 12-12, these same equations mistakenly
   switch the fiber vectors e1 and e2(which are called "a" and "b" in ANSYS).  The first time these
   equations appear are in 12.5.1, 12.5.2, 12.5.3.  They also appear incorrectly like this later.
*/

/*
	    printf( "\n\n              These are the updated coordinates \n");
	    printf( "\n           x            y               z\n");

	    for( i = 0; i < numnp; ++i )
	    {
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i)*(*(U+ndof*i+4));
		fpointx = *(coord+nsd*i) + *(U+ndof*i) +
		    *(Uz_fib+i)*fdum;
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 1)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i + 1)*(*(U+ndof*i+4));
		fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1) +
		    *(Uz_fib+i)*fdum;
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 2)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i + 2)*(*(U+ndof*i+4));
		fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2) +
		    *(Uz_fib+i)*fdum;
		printf("\n node %3d %14.9f %14.9f %14.9f",i,fpointx,fpointy,fpointz);
	    }
	    printf(" \n");
*/

	    check = shwriter( bc, connect, coord, el_matl, force, id, fiber_xyz,
		matl, name, strain, strain_node, stress, stress_node, U, Uz_fib);
	    if(!check) printf( " Problems with shwriter \n");
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

	    check = shLanczos(A, bc, connect, coord, eigen, el_matl, id, idiag, K_diag,
		lamina_ref, fiber_xyz, mass, matl, ritz, num_eigen);
	    if(!check) printf( " Problems with shLanczos \n");

/* Write out the eigenvalues to a file */

	    check = eval_data_print( eigen, name, nmode);
	    if(!check) printf( " Problems with eval_data_print \n");

/* Write out the eigenmodes to a (name).mod-x.osh file */

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

		check = shKassemble(A, connect, coord, el_matl, fiber_vec,
			vector_dum, id, idiag, K_diag, lm, lamina_ref, fiber_xyz, matl,
			node_counter, strain, strain_node, stress, stress_node, U);
		if(!check) printf( " Problems with shKassembler \n");

/* Calcuate the local z fiber displacement on each node */

		for( i = 0; i < numnp; ++i )
		{
/* Calcuate the projection of displacement vector to local z fiber vector */

		    *(node_vec) = *(coord+nsd*(i+numnp))-*(coord+nsd*i);
		    *(node_vec+1) = *(coord+nsd*(i+numnp)+1)-*(coord+nsd*i+1);
		    *(node_vec+2) = *(coord+nsd*(i+numnp)+2)-*(coord+nsd*i+2);
		    fdum = *(node_vec)*(*(node_vec)) +
			*(node_vec+1)*(*(node_vec+1)) + *(node_vec+2)*(*(node_vec+2));
		    fdum = sqrt(fdum);
		    *(node_vec) /= fdum;
		    *(node_vec+1) /= fdum;
		    *(node_vec+2) /= fdum;
		    check = dotX( (Uz_fib+i), (node_vec), (U+ndof*i), nsd);
		    if(!check) printf( " Problems with dotX \n");
		    /* *(Uz_fib+i) *= .5;*/
		    /*printf("\n node %3d %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e",
			i,*(Uz_fib+i),*(U+ndof*i),*(U+ndof*i+1),*(U+ndof*i+2),
			*(node_vec), *(node_vec+1), *(node_vec+2),fdum);*/
		}

		check = shwriter( bc, connect, coord, el_matl, force, id, fiber_xyz,
		    matl, name_mode, strain, strain_node, stress, stress_node, U, Uz_fib);
		if(!check) printf( " Problems with shwriter \n");

		++counter;

	    }
	}

/* Calculating the value of the Volumes */

	check = shVolume( connect, coord, Voln);
	if(!check) printf( " Problems with shVolume \n");

	/*printf("\nThis is the Volume\n");
	for( i = 0; i < numel; ++i )
	{
		printf("%4i %12.4e\n",i, *(Voln + i));
	}*/

	timec = clock();
	printf("\n\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(mem_SDIM);
	free(matl);
	free(mem_sh_double);
	free(mem_rotate_double);
	free(mem_double);
	free(mem_int);
	free(mem_XYZPhiI);
	if(modal_flag) free(eigen);
	free(A);
}


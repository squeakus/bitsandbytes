/*
    This program performs finite element analysis on a quad by reading in
    the data, assembling the P vector, and then solving the system using
    dynamic relaxation or the conjugate gradient method.  This element
    is capable of handling large deformation loads.  The material is
    assumed to be hypo-elastic and the stress is updated using the
    Jaumann Stress Rate.

	        Udated 10/27/09

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <unistd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../quad/qdconst.h"
#include "../quad/qdstruct.h"
#include "../../common/gnuplot2.h"

#define flag        3                     /* flag for postprocessor */
#define SMALL      1.e-20

int GnuplotWriter2( char *, FORCE_DAT, double );

int qdKassemble(double *, int *, double *, int *, double *, int *, int *,
	double *, int *, double *, MATL *, double *, STRAIN *, SDIM *, STRESS *,
	SDIM *, double *);

int qdArea( int *, double *, double *);

int qdwriter ( BOUND , int *, double *, int *, double *, int *, MATL *,
	char *, STRAIN *, SDIM *, STRESS *, SDIM *, double *);

int matX(double *,double *,double *, int ,int ,int );

int Boundary( double *, BOUND );

int qdConjPassemble(double *, int *, double *, int *, double *, MATL *,
	double *, double *);

int qdPassemble(int *, double *, double *, int *, MATL *, double *, STRESS *,
	double * );

int qd2Passemble(int *, double *, double *, int *, double *, double *, MATL *,
	double *, STRESS *, double *, double *, double * );

int qdFMassemble(int *, double *, double *, int *, double *, double *, double *,
	double *, MATL *, double *);

int formid( BOUND, int *);

int local_vectors_old( int *, double *, double * );

int local_vectors( int *, double *, double * );

int qdshg( double *, int, double *, double *, double *);

int qdreader( BOUND , int *, double *, int *, double *, MATL *, char *, FILE *,
	STRESS *, SDIM *, double *);

int Memory( double **, int, int **, int, MATL **, int , XYZI **, int,
	SDIM **, int, STRAIN **, STRESS **, int );

int qdshl(double, double *, double * );

int analysis_flag, dof, sdof, neqn, nmat, nmode, numnp, numel, plane_stress_flag,
	sof, flag_3D, local_vec_flag;
int stress_read_flag, element_stress_read_flag, element_stress_print_flag,
	gauss_stress_flag, dynamic_relax_flag, stress_update_flag;

int static_flag;
int Passemble_flag, Passemble_CG_flag;

int LU_decomp_flag, numel_K, numel_P;

double shg[sosh], shgh[sosh], shg_node[sosh], shl[sosh],shl_node[sosh], shl_node2[sosh_node2],
	*Area0, w[num_int];
double dt, damp_const;
int iteration, iteration_max;
double tolerance;

int main(int argc, char** argv)
{
	int i, j, k, i2, i3;
	int *id, check, name_length, counter, counter2, MemoryCounter, dum, one_step;
	int iter_conj, iter_conj2, iter_conj_extra;
	double  d_iteration_max, iteration_const;
	int matl_num;
	double Emod, G, Gt, thickness, K, Pois;
	XYZI *mem_XYZI;
	int *mem_int, sofmi, sofmf, sofmSTRESS, sofmXYZI, sofmSDIM, ptr_inc;
	MATL *matl;
	double *mem_double;
	double fpointx, fpointy, fpointz;
	int *connect, *el_matl, node;
	double *coord, *coord0, *force, *P_global, *U, *dU, *V, *dUm1,
		*Arean, *mass, *node_counter, *local_xyz, *localh_xyz;
	double coord_el_trans[neqel], *coordh;
	char name[30], buf[ BUFSIZ ], *ccheck;
	FILE *o1, *o2, *o3, *o4;
	double det[num_int], wXdet;
	BOUND bc;
	STRESS *stress;
	STRAIN *strain;
	SDIM *stress_node, *strain_node, *mem_SDIM;
	double xxaddyy, xxsubyy, xysq;
	double g;
	double hydro;
	long timec;
	long timef;
	double bet, alp, timer;
	double RAM_max, RAM_usable, RAM_needed, MEGS;
	int surf_el_counter, surface_el_flag;

	double *mem_double2;
	double *p, *P_global_CG, *r, *z, *B;
	double alpha, alpha2, beta;
	double fdum, fdum2;
	int sofmf2;

	char gplot_dat_name[30];
	FILE *gplot_dat;
	FORCE_DAT max;

/* The variables below are not really used.  They are only needed because
   some of the subroutines I use are from the linear quad code, which
   have these variables as part of their argument list.
*/
	double A[1], K_diag[1];
	int lm[1], idiag[1];

	sof = sizeof(double);

/* Create local shape funcions */

	g = 2.0/sq3;
	check = qdshl(g, shl, w );
	if(!check) printf( "Problems with qdshl \n");

	memset(name,0,30*sizeof(char));
	
	printf("What is the name of the file containing the \n");
	printf("quad structural data? (example: comb-hill-back)\n");
	scanf( "%30s",name);

/*   o1 contains all the structural data */
/*   o2 contains input parameters */
/*   o4 contains dynamic data   */

	o1 = fopen( name,"r" );
	o2 = fopen( "qdinput","r" );
	o3 = fopen( "data","w" );
	o4 = fopen( "nqdinput","r" );

	if(o1 == NULL ) {
		printf("Can't find file %30s\n",name);
		exit(1);
	}

	if( o2 == NULL ) {
		printf("Can't find file qdinput\n");
		tolerance = 1.e-13;
		iteration_max = 2000;
		RAM_max = 160.0;
		element_stress_read_flag = 0;
		element_stress_print_flag = 0;
		gauss_stress_flag = 1;
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
	gauss_stress_flag = 1;

	if( o4 == NULL ) {
		printf("Can't find file nqdinput\n");
		exit(1);
	}
	else
	{
		fgets( buf, BUFSIZ, o4 );
		fscanf( o4, "%d\n ", &dynamic_relax_flag);
		fgets( buf, BUFSIZ, o4 );
		fscanf( o4, "%d\n ", &iteration_max);
		fgets( buf, BUFSIZ, o4 );
		fscanf( o4, "%lf\n ", &dt);
		fgets( buf, BUFSIZ, o4 );
		fscanf( o4, "%lf\n ", &damp_const);
		fgets( buf, BUFSIZ, o4 );
		fscanf( o4, "%d\n ", &stress_update_flag);
	}


	fgets( buf, BUFSIZ, o1 );
	fscanf( o1, "%d %d %d %d %d\n ",
		&numel,&numnp,&nmat,&nmode,&plane_stress_flag);
	dof=numnp*ndof;
	sdof=numnp*nsd;
	static_flag = 1;

	numel_P = numel;
	numel_K = 0;
	LU_decomp_flag = 0;

	d_iteration_max = (double) iteration_max;
	iteration_const = 1.0/(d_iteration_max*dt);

/* These variables are for when the Conjugate Gradient method is used.  There
   are 2 loops, with "inter_conj" the outer loop, and the inner loop equal
   to 5 except when iteration_max%5 != 0.  When iteration_max%5 != 0, then
   the last inner steps, occurring at the last outer step, equal iteration_max%5
   and the total steps will be iteration_max.

   I am inclined to only to loop over multiples of 5 and not worry about the
   extra steps when iteration_max is not a multiple of 5.  But I thought of an
   easy way to do it, and this may prove to be a useful example implementation
   in the future for when this is important.
*/
 
	iter_conj_extra = iteration_max%5;
	one_step = 0;
	if(iter_conj_extra) one_step = 1;
	dum = iteration_max - iter_conj_extra;
	iter_conj2 = (int)(dum/5);

/*   Begin allocation of meomory */

	MemoryCounter = 0;

/* For the doubles */
	sofmf=3*sdof + 7*dof + 2*numel + numnp + 2*numel*npel*nsdsq;
	MemoryCounter += sofmf*sizeof(double);
	printf( "\n Memory requrement for doubles is %15d bytes\n",MemoryCounter);

/* For the integers */
	sofmi=numel*npel + dof + numel + numnp+1;
	MemoryCounter += sofmi*sizeof(int);
	printf( "\n Memory requrement for integers is %15d bytes\n",MemoryCounter);

/* For the XYZI integers */
	sofmXYZI=numnp+1;
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

	if(!check) printf( "Problems with Memory \n");

/* For the doubles */
	                                        ptr_inc=0; 
	coord0=(mem_double+ptr_inc);            ptr_inc += sdof;
	coord=(mem_double+ptr_inc);             ptr_inc += sdof;
	coordh=(mem_double+ptr_inc);            ptr_inc += sdof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	dU=(mem_double+ptr_inc);                ptr_inc += dof;
	dUm1=(mem_double+ptr_inc);              ptr_inc += dof;
	V=(mem_double+ptr_inc);                 ptr_inc += dof;
	mass=(mem_double+ptr_inc);              ptr_inc += dof;
	P_global=(mem_double+ptr_inc);          ptr_inc += dof;
	Arean=(mem_double+ptr_inc);             ptr_inc += numel; 
	Area0=(mem_double+ptr_inc);             ptr_inc += numel;
	node_counter=(mem_double+ptr_inc);      ptr_inc += numnp; 
	local_xyz=(mem_double+ptr_inc);         ptr_inc += numel*npel*nsdsq;
	localh_xyz=(mem_double+ptr_inc);        ptr_inc += numel*npel*nsdsq;

/* For the integers */
	                                        ptr_inc = 0; 
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npel; 
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
        el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
        bc.force =(mem_int+ptr_inc);            ptr_inc += numnp;
        bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;

/* For the XYZI integers */
	                                  ptr_inc = 0; 
	bc.fix =(mem_XYZI+ptr_inc);       ptr_inc += numnp;
	bc.num_fix=(mem_XYZI+ptr_inc);    ptr_inc += 1;

/* For the SDIM doubles */
                                                ptr_inc = 0;
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp;
	strain_node=(mem_SDIM+ptr_inc);         ptr_inc += numnp;

	timec = clock();
	timef = 0;

	for( k = 0; k < numel; ++k )
	{
		for( j = 0; j < num_int; ++j )
		{
			stress[k].pt[j].xx=.0;
			stress[k].pt[j].yy=.0;
			stress[k].pt[j].xy=.0;
		}
	}

	stress_read_flag = 1;
	check = qdreader( bc, connect, coord0, el_matl, force, matl, name, o1,
		stress, stress_node, U);
	if(!check) printf( "Problems with reader \n");

	local_vec_flag = 1;
#if 0
	local_vec_flag = 0;
#endif

	if(local_vec_flag)
	{
	    check = local_vectors( connect, coord0, local_xyz );
	    if(!check) printf( " Problems with local_vectors \n");

	    check = local_vectors( connect, coord0, localh_xyz );
	    if(!check) printf( " Problems with local_vectors \n");
	}
	else
	{
	    check = local_vectors_old( connect, coord0, local_xyz );
	    if(!check) printf( " Problems with local_vectors_old \n");

	    check = local_vectors_old( connect, coord0, localh_xyz );
	    if(!check) printf( " Problems with local_vectors_old \n");
	}

	printf(" \n\n");

	printf( "\n The total memory requrement is %15d bytes\n", MemoryCounter);
	MEGS = ((double)(MemoryCounter))/MB;
	printf( "\n Which is %16.4e MB\n", MEGS);

/* Because this is a dynamic code, the id array is not needed except for
   the qdwriter subroutine */

	check = formid( bc, id );
	if(!check) printf( "Problems with formid \n");

	for( i = 0; i < dof; ++i )
	{
/* Initializing the data for change in displacement */

		 *(dU+i)=.0;
		 *(dUm1+i)=.0;

/* Initializing the data for velocity */

		*(V+i)=0.0;
	}

	for( i = 0; i < sdof; ++i )
	{

/* Initializing the data for x,y,z coordindates */

		*(coord+i)=*(coord0+i);
		*(coordh+i)=*(coord0+i);
	}

/* Initializing the data for the Area */

	check = qdArea( connect, coord0, Area0);
	if(!check) printf( "Problems with qdArea \n");

	printf( "\n\n This is the Area matrix \n");
/*
	for( k = 0; k < numel; ++k )
	{
		printf(" \n");
		printf("Area for element %d %14.5f ", k, *(Area0+k));
	}
	printf("\n\n");
*/

/* The mass matrix */

	printf(" \n");
	printf("mass  matrix\n");

	check = qdFMassemble(connect, coord, coordh, el_matl, force, mass, local_xyz,
		localh_xyz, matl, U );
	if(!check) printf( "Problems with qdFMassemble \n");

	printf( "\n\n This is the mass matrix \n");
	for( i = 0; i < dof; ++i )
	{
	    *(mass + i) += SMALL;
	    printf( "\ndof %5d   %16.8e %16.8e",i,*(mass+i), *(force+i));
	}
	printf(" \n");

	printf(" \n");

/* For the doubles */
	sofmf2 = 5*dof;
	mem_double2=(double *)calloc(sofmf2,sizeof(double));

	if(!mem_double2 )
	{
		printf( "failed to allocate memory for doubles\n ");
		exit(1);
	}

/* For the Conjugate Gradient Method doubles */

	                                         ptr_inc = 0;
	p=(mem_double2+ptr_inc);                 ptr_inc += dof;
	P_global_CG=(mem_double2+ptr_inc);       ptr_inc += dof;
	r=(mem_double2+ptr_inc);                 ptr_inc += dof;
	z=(mem_double2+ptr_inc);                 ptr_inc += dof;
	B=(mem_double2+ptr_inc);                 ptr_inc += dof;

/* Using Conjugate gradient method to find displacements */

	memset(P_global_CG,0,dof*sof);
	memset(p,0,dof*sof);
	memset(r,0,dof*sof);
	memset(z,0,dof*sof);

/* Look for the DOFs of the largest and 2nd largest point load forces.  The corresponding
   DOFs of P_global or P_global_CG will be plotted over time against this force (depending
   on which method is used) to see if convergence has been reached.
*/

	for( i = 0; i < dof ; ++i )
	{
		if( fabs(max.force) < fabs(*(force + i)) )
		{
			max.force = *(force + i);
			max.dof_force = i;
			printf(" %f %f \n", max.force, *(force + i));
		}
	}
	for( i = 0; i < dof ; ++i )
	{
		if( fabs(max.force2) < fabs(*(force + i)) && i != max.dof_force )
		{
			max.force2 = *(force + i);
			max.dof_force2 = i;
		}
	}

	printf( "                    2nd max       max      2nd max           max\n");
	printf( "force                 %5d     %5d   %14.6e %14.6e\n", max.dof_force2,
		max.dof_force, max.force2, max.force);

/* Open a file with the name (name).gplot that will contain either P_global or P_global_CG
   over time as well as the point forces.  */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(gplot_dat_name,0,30*sizeof(char));

	ccheck = strncpy(gplot_dat_name, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(gplot_dat_name+name_length, ".gplot", 6);
	if(!ccheck) printf( " Problems with strncpy \n");

	gplot_dat = fopen( gplot_dat_name,"w" );


/* Dynamic Relaxation occurs below */
/* The code below uses qd2Passemble which combines qdPassemble and
   qdConjPassemble.  I really only need the functionality of qdPassemble
   and not qdConjPassemble to calculate P_global, so it should have been
   more efficient to only call qdPassemble, but an analysis using the
   profiling tool shows this is not the case.
*/

	if( dynamic_relax_flag )
	{

/* The variables below were named based on the paper:
*/
	    fdum = damp_const*dt;
	    bet = (2.0 - fdum)/(2.0 + fdum);
	    alp = 2.0*dt*dt/(2.0 + fdum);
	    timer = 0.0;

	    counter2 = 0;
	    printf("\n vvvv \n");
	    Passemble_flag = 1;
	    Passemble_CG_flag = 0;
	    fprintf(o3, " %d %d \n",iteration_max,1);
	    for( iteration = 0; iteration < iteration_max; ++iteration )
	    {
		timer=timer+dt;
		/*fprintf(o3, " %f %f \n",timer,*(dU+dof-1)/numel*10);*/
		printf( " \n %3d \n",iteration);

/*
		printf("\n %5d  %16.8e %14.6f %14.6f %14.6f",
			 iteration, timer, *(P_global+812), *(P_global+815), *(P_global+818));
		printf("\n %5d  %16.8e %14.6f %14.6f %14.6f",
			 iteration, timer, *(P_global+2), *(P_global+3), *(P_global+8));
*/

/*
		check = qdPassemble( connect, coord, coordh, el_matl, matl, P_global,
		    stress, dU);
		if(!check) printf( "Problems with qdPassemble \n");
*/
		check = qd2Passemble( connect, coord, coordh, el_matl, local_xyz, localh_xyz,
		    matl, P_global, stress, dU, P_global_CG, p);
		if(!check) printf( "Problems with qd2Passemble \n");

		fprintf( gplot_dat, "%5d  %16.8e %14.6e %14.6e %14.6e %14.6e\n",
		    iteration, counter2*dt, *(P_global+max.dof_force),
		    *(P_global+max.dof_force2), max.force2, max.force);
		if( fabs(max.P_global) < fabs(*(P_global + max.dof_force)) )
		    max.P_global = *(P_global + max.dof_force);
		if( fabs(max.P_global2) < fabs(*(P_global + max.dof_force2)) )
		    max.P_global2 = *(P_global + max.dof_force2);

		/*for( j = 0; j < dof; ++j )
		{
		    printf("\n vvvv  %3d %12.10f  %12.10f %14.5f",j,*(dU+j),
			*(V+j),*(P_global+j));
		}*/

		for( j = 0; j < numnp; ++j )
		{
#if 0
		    for( i2 = 0; i2 < ndof; ++i2 )
		    {
			/*printf("\n  %3d %12.10f  %12.10f %14.5f",
				ndof*j+i2,*(dU+ndof*j+i2),
				*(V+ndof*j),*(P_global+ndof*j+i2));*/
			/*printf( "%4d %f %f %f %f\n",ndof*j+i2,
				*(P_global+ndof*j+i2),*(dU+ndof*j+i2),
				*(V+ndof*j),*(coord+nsd*j));*/
		    }
#endif

/* Calculate the displacement increment, dU.  Read the notes in this directory
   to see how the expression below was derived. */

		    *(dU+ndof*j) = bet*(*(dUm1+ndof*j)) - alp*( *(P_global+ndof*j) -
			*(force+ndof*j) )/(*(mass+ndof*j));
		    *(dU+ndof*j+1) = bet*(*(dUm1+ndof*j+1)) - alp*( *(P_global+ndof*j+1) -
			*(force+ndof*j+1) )/(*(mass+ndof*j+1));
		    *(dU+ndof*j+2) = bet*(*(dUm1+ndof*j+2)) - alp*( *(P_global+ndof*j+2) -
			*(force+ndof*j+2) )/(*(mass+ndof*j+2));

/* Update of the XYZ coordinate matrix  */

		    *(coordh+nsd*j) = *(coord+nsd*j)+.5*(*(dU+ndof*j));
		    *(coordh+nsd*j+1) = *(coord+nsd*j+1)+.5*(*(dU+ndof*j+1));
		    *(coordh+nsd*j+2) = *(coord+nsd*j+2)+.5*(*(dU+ndof*j+2));

		    *(coord+nsd*j) += *(dU+ndof*j);
		    *(coord+nsd*j+1) += *(dU+ndof*j+1);
		    *(coord+nsd*j+2) += *(dU+ndof*j+2);

/* Update of the Velocity matrix  */

		    *(V+ndof*j) = *(dU+ndof*j)/dt;
		    *(V+ndof*j+1) = *(dU+ndof*j+1)/dt;
		    *(V+ndof*j+2) = *(dU+ndof*j+2)/dt;

/* Set old displacement increment, dUm1, to dU  */

		    *(dUm1+ndof*j) = *(dU+ndof*j);
		    *(dUm1+ndof*j+1) = *(dU+ndof*j+1);
		    *(dUm1+ndof*j+2) = *(dU+ndof*j+2);

		}

/*
   To handle prescribed displacements, I add each displacement
   incrementally to their corresponding nodes based on the following:

    timer = iteration*dt

	 V = U_fixed/(iteration_max*dt)
     coord = coord0 + [U_fixed/(iteration_max*dt)]*timer
    coordh = coord0 + [U_fixed/(iteration_max*dt)]*(timer - dt/2.0)
*/
		for( i2 = 0; i2 < bc.num_fix[0].x; ++i2 )
		{
		    node = bc.fix[i2].x;
		    *(dU+ndof*node) = *(U+ndof*node)/d_iteration_max;
		    *(V+ndof*node) = *(U+ndof*node)*iteration_const;
		    *(coord+nsd*node) = *(coord0+nsd*node) +
			*(U+ndof*node)*iteration_const*timer;
		    *(coordh+nsd*node) = *(coord0+nsd*node) +
			*(U+ndof*node)*iteration_const*(timer - dt/2.0);
		}
		for( i2 = 0; i2 < bc.num_fix[0].y; ++i2 )
		{
		    node = bc.fix[i2].y;
		    *(dU+ndof*node+1) = *(U+ndof*node+1)/d_iteration_max;
		    *(V+ndof*node+1) = *(U+ndof*node+1)*iteration_const;
		    *(coord+nsd*node+1) = *(coord0+nsd*node+1) +
			*(U+ndof*node+1)*iteration_const*timer;
		    *(coordh+nsd*node+1) = *(coord0+nsd*node+1) +
			*(U+ndof*node+1)*iteration_const*(timer - dt/2.0);
		}
		for( i2 = 0; i2 < bc.num_fix[0].z; ++i2 )
		{
		    node=bc.fix[i2].z;
		    *(dU+ndof*node+2) = *(U+ndof*node+2)/d_iteration_max;
		    *(V+ndof*node+2) = *(U+ndof*node+2)*iteration_const;
		    *(coord+nsd*node+2) = *(coord0+nsd*node+2) +
			*(U+ndof*node+2)*iteration_const*timer;
		    *(coordh+nsd*node+2) = *(coord0+nsd*node+2) +
			*(U+ndof*node+2)*iteration_const*(timer - dt/2.0);
		    /*printf("\n vvvv  %3d %3d %12.10f  %12.10f %14.5f %14.5f",i2,iteration,
			*(coord+nsd*node+2), *(coordh+nsd*node+2), *(U+ndof*node+2),
			iteration_const);*/
		}

		++counter2;

	    }

	    /*for( j = 0; j < dof; ++j )
	    {
		printf("\n vvvv  %5d %16.8e %14.6f %14.6f",j,*(U+j), *(force+j),*(P_global+j));
	    }*/
	}


/* The Conjugate Gradient Method is used below */
/* The code below uses qd2Passemble which combines qdPassemble and
   qdConjPassemble.  There is also an additional
   loop over i outside the k loop which I have added to
   possibly mimic the resetting of the initial guess step
   as given in the paper:

	Stranden, I. and M. Lidauer, Solving Large Mixed Linear Models Using
	Preconditioned Conjugate Gradient Iteration, Journal of Dairy Science,
	Vol. 82 No. 12, 1999, page 2779-2787.

 */

	iter_conj = 5;
	if( !dynamic_relax_flag )
	{

	    for( j = 0; j < dof; ++j )
	    {
		*(mass + j) /= 100.0;
		*(B + j) = *(force + j);
		*(r + j) = *(force + j);
	    }

	    printf("\n vvvv \n");

	    counter2 = 0;
	    for( i3 = 0; i3 < iter_conj2 + one_step; ++i3 )
	    {
		for( j = 0; j < dof; ++j )
		{
		    /**(r+j) = *(force +j) - *(P_global + j);
		    *(r + j) = *(B + j) - *(P_global_CG + j);*/

		    *(z + j) = *(r + j)/(*(mass + j));
		    *(p+j) = *(z+j);

		    *(dU+j) = 0.0;
		    *(V+j) = 0.0;
		}

		check = Boundary (r, bc);
		if(!check) printf( "Problems with Boundary \n");

		check = Boundary (p, bc);
		if(!check) printf( "Problems with Boundary \n");

		alpha = 0.0;
		alpha2 = 0.0;
		beta = 0.0;
		fdum2 = 1000.0;
		counter = 0;
		check = dotX(&fdum, r, z, dof);

/*
   Within qd2Passemble, stress is updated, but there is no need to caculate
   P_global until the while loop is finished.
*/
		Passemble_flag = 0;
		Passemble_CG_flag = 1;

		/*printf("\n iteration %3d iteration max %3d \n", iteration, iteration_max);*/
		/*for( iteration = 0; iteration < iteration_max; ++iteration )*/

		if( i3 == iter_conj2 ) iter_conj = iter_conj_extra; 
		while(fdum2 > tolerance && counter < iter_conj )
		{
		    printf( "\n %3d %3d %16.8e\n", counter2, counter, fdum2);

		    /*printf("\n %5d  %16.8e %14.6f %14.6f %14.6f",
			 counter2, counter2*dt, *(P_global_CG+33), *(P_global_CG+34), *(P_global_CG+35));*/

		    if(stress_update_flag)
		    {
			check = qd2Passemble( connect, coord, coordh, el_matl, local_xyz, localh_xyz,
			    matl, P_global, stress, dU, P_global_CG, p);
			if(!check) printf( "Problems with qd2Passemble \n");
		    }
		    else
		    {
			check = qdConjPassemble( A, connect, coord, el_matl, local_xyz, matl,
			    P_global_CG, p);
			if(!check) printf( "Problems with qdConjPassemble \n");
		    }

/* The reason qdPassemble was called(before I wrote "qd2Passemble" is to do stress updating rather
   than the calculation of P_global.
*/
/*
		    check = qdPassemble( connect, coord, coordh, el_matl, matl, P_global,
			stress, dU);
		    if(!check) printf( "Problems with qdPassemble \n");
*/

		    /*printf( "\n %3d %16.8e %16.8e\n",counter, *(P_global + 44), *(P_global_CG + 44));*/

		    check = Boundary (P_global_CG, bc);
		    if(!check) printf( "Problems with Boundary \n");
		    check = dotX(&alpha2, p, P_global_CG, dof);
		    alpha = fdum/(SMALL + alpha2);
		    for( j = 0; j < dof; ++j )
		    {
			/*printf( "%4d %14.5e  %14.5e  %14.5e  %14.5e  %14.5e %14.5e\n",j,alpha,
			    beta,*(U+j),*(r+j),*(P_global_CG+j),*(p+j));*/
			*(U+j) += alpha*(*(p+j));
			*(r+j) -= alpha*(*(P_global_CG+j));
			/**(r+j) = *(force + j) - *(P_global+j);*/
			*(z + j) = *(r + j)/(*(mass + j));

			*(dU+j) = alpha*(*(p+j));
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
			    *(U+j)*bet,*(r+j)*bet,*(P_global_CG+j)*alp/(*(mass+j)),
			*(force+j)*alp/(*(mass+j)));*/
			*(p+j) = *(z+j)+beta*(*(p+j));
		    }
		    check = Boundary (p, bc);
		    if(!check) printf( "Problems with Boundary \n");

		    for( j = 0; j < numnp; ++j )
		    {

/* Update of the XYZ coordinate matrix  */

			*(coordh+nsd*j) = *(coord+nsd*j)+.5*(*(dU+ndof*j));
			*(coordh+nsd*j+1) = *(coord+nsd*j+1)+.5*(*(dU+ndof*j+1));
			*(coordh+nsd*j+2) = *(coord+nsd*j+2)+.5*(*(dU+ndof*j+2));

			*(coord+nsd*j) += *(dU+ndof*j);
			*(coord+nsd*j+1) += *(dU+ndof*j+1);
			*(coord+nsd*j+2) += *(dU+ndof*j+2);

/* Update of the Velocity matrix  */

			*(V+ndof*j) = *(dU+ndof*j)/dt;
			*(V+ndof*j+1) = *(dU+ndof*j+1)/dt;
			*(V+ndof*j+2) = *(dU+ndof*j+2)/dt;
		    }

		    for( i2 = 0; i2 < bc.num_fix[0].x; ++i2 )
		    {
			node = bc.fix[i2].x;
			*(dU+ndof*node) = *(U+ndof*node)/d_iteration_max;
			*(V+ndof*node) = *(U+ndof*node)*iteration_const;
			*(coord+nsd*node) = *(coord0+nsd*node) +
			    *(U+ndof*node)*iteration_const*timer;
			*(coordh+nsd*node) = *(coord0+nsd*node) +
			    *(U+ndof*node)*iteration_const*(timer - dt/2.0);
		    }
		    for( i2 = 0; i2 < bc.num_fix[0].y; ++i2 )
		    {
			node = bc.fix[i2].y;
			*(dU+ndof*node+1) = *(U+ndof*node+1)/d_iteration_max;
			*(V+ndof*node+1) = *(U+ndof*node+1)*iteration_const;
			*(coord+nsd*node+1) = *(coord0+nsd*node+1) +
			    *(U+ndof*node+1)*iteration_const*timer;
			*(coordh+nsd*node+1) = *(coord0+nsd*node+1) +
			    *(U+ndof*node+1)*iteration_const*(timer - dt/2.0);
		    }
		    for( i2 = 0; i2 < bc.num_fix[0].z; ++i2 )
		    {
			node = bc.fix[i2].z;
			*(dU+ndof*node+2) = *(U+ndof*node+2)/d_iteration_max;
			*(V+ndof*node+2) = *(U+ndof*node+2)*iteration_const;
			*(coord+nsd*node+2) = *(coord0+nsd*node+2) +
			    *(U+ndof*node+2)*iteration_const*timer;
			*(coordh+nsd*node+2) = *(coord0+nsd*node+2) +
			    *(U+ndof*node+2)*iteration_const*(timer - dt/2.0);
		    }

		    ++counter; ++counter2;

		    if( fdum2 < tolerance ) i3 = iter_conj2 + 2;

		    if(counter > 90000 )
		    {
			printf( " %15.6e \n",fdum2);
			break;
		    }
		}

		Passemble_flag = 0;
		Passemble_CG_flag = 1;

		if(stress_update_flag)
		{
		    check = qd2Passemble( connect, coord, coordh, el_matl, local_xyz, localh_xyz,
			matl, P_global, stress, dU, P_global_CG, U);
		    if(!check) printf( "Problems with qd2Passemble \n");
		}
		else
		{
		    check = qdConjPassemble( A, connect, coord, el_matl, local_xyz, matl,
			P_global_CG, U);
		    if(!check) printf( "Problems with qdConjPassemble \n");
		}
/*
		check = qdPassemble( connect, coord, coordh, el_matl, matl, P_global,
		    stress, dU);
		if(!check) printf( "Problems with qdPassemble \n");
*/

		fprintf( gplot_dat, "%5d  %16.8e %14.6e %14.6e %14.6e %14.6e\n",
		    iteration, counter2*dt, *(P_global_CG+max.dof_force),
		    *(P_global_CG+max.dof_force2), max.force2, max.force);
		if( fabs(max.P_global_CG) < fabs(*(P_global_CG + max.dof_force)) )
		    max.P_global_CG = *(P_global_CG + max.dof_force);
		if( fabs(max.P_global_CG2) < fabs(*(P_global_CG + max.dof_force2)) )
		    max.P_global_CG2 = *(P_global_CG + max.dof_force2);

		for( i = 0; i < dof; ++i )
		{
/* I have tried calculting *(r + i) based on:

                    *(r + i) = *(force + i) - *(P_global + i);

   to again find balance between the force and P_global using stresses based on
   stress updating rather than a total Lagrangian approach.  This method did not
   work and the solution did not converge.  The way it is done now is with the
   residual based on total Lagrangian calculation of P_global_CG which works much
   better.

   I also tried: 

                    *(r + i) = *(B + i) - *(P_global_CG + i);

   which also didn't work or converge.  The above is similar to what I did when
   I solved the Poission equation using the Conjugate Gradient method in SLFCFD.
   But there are significant differences between what I am doing here and what
   is being done in SLFCFD, so I shouldn't expect to implement it in exactly the
   same manner.
*/
		    *(r + i) = *(force + i) - *(P_global_CG + i);
		    *(B + i) = *(P_global_CG + i);
		}
		/*if(fdum3 < 1.0e-8) break;
		printf( "\n fffff  %16.8e\n", fdum3);*/

		if(counter > iteration_max - 1 )
		{
		    printf( "\nMaximum iterations %4d reached.  Residual is: %16.8e\n",
			counter,fdum2);
		    printf( "Problem may not have converged during Conj. Grad.\n");
		}
	    }

	    /*for( j = 0; j < dof; ++j )
	    {
		printf("\n vvvv  %5d %16.8e %14.6f %14.6f",j,*(r+j), *(force+j),*(P_global_CG+j));
	    }*/

/*
The lines below are for testing the quality of the calculation:

1) r should be 0.0
2) P_global( = A*U ) - force should be 0.0
*/

/*
	    check = qdConjPassemble( A, connect, coord0, el_matl, local_xyz, matl,
		P_global_CG, U);
	    if(!check) printf( "Problems with qdConjPassemble \n");

	    for( j = 0; j < dof; ++j )
	    {
		printf( "%4d %14.6f  %14.6f %14.6f %14.6f   %14.6f %14.6f %14.6f\n",j,alpha,beta,
			*(U+j),*(r+j),*(P_global_CG+j),*(P_global+j),*(force+j));
	    }
*/
	    /*for( j = 0; j < dof; ++j )
	    {
		    *(coord + j) = *(coord0 + j) + *(U + j);
	    }*/

	    printf( "\n %3d %16.8e\n", counter2, fdum2);
	    /*exit(1);*/

	}

/* Calculating the final value of the Areas */

	check = qdArea( connect, coord, Arean);
	if(!check) printf( "Problems with qdArea \n");

/*
	printf("\nThis is the Area\n");
	for( i = 0; i < numel; ++i )
	{
		printf("%4i %12.4e\n",i, *(Arean + i));
	}
*/

/*
	printf(" \n\n");
	for( i = 0; i < numnp; i+=4 )
	{
		printf("\n %3d	%7.4f    %7.4f    %7.4f",i+1,
			*(coord+nsd*i), *(coord+nsd*i+1),*(coord+nsd*i+2));
		printf("\n %3d	%7.4f    %7.4f    %7.4f",i+2,
			*(coord+nsd*(i+3)), *(coord+nsd*(i+3)+1),*(coord+nsd*(i+3)+2));
		printf("\n %3d	%7.4f    %7.4f    %7.4f", i+3,
			*(coord+nsd*(i+1)), *(coord+nsd*(i+1)+1),*(coord+nsd*(i+1)+2));
		printf("\n %3d	%7.4f    %7.4f    %7.4f",i+4,
			*(coord+nsd*(i+2)), *(coord+nsd*(i+2)+1),*(coord+nsd*(i+2)+2));
	}
	printf(" \n\n");
*/
	if(!dynamic_relax_flag && !stress_update_flag)
	{
	    for( k = 0; k < numel; ++k )
	    {
		for( j = 0; j < num_int; ++j )
		{

/* Count the number of times a particular node is part of an element */

		    node = *(connect+npel*k+j);

		    *(node_counter + node) += 1.0;
		}
	    }

/* Determine the stresses and strains based on a total Lagrangian calculation.  Since
   I only want the reaction forces based on fixed displacements to update the "force"
   variable, so I use "p", one of the varibles used in the Conjugate Gradient method
   to get the reaction forces.
 */

	    analysis_flag = 2;
	    memset(p,0,dof*sof);
	    check = qdKassemble(A, connect, coord, el_matl, p, id, idiag, K_diag,
		lm, local_xyz, matl, node_counter, strain, strain_node, stress,
		stress_node, U);
	    if(!check) printf( " Problems with qdKassembler \n");

/* Set reaction forces */

	    for( i = 0; i < numnp; ++i )
	    {
		if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0 )
		{
		    *(force+ndof*i) = *(p+ndof*i);
		    *(force+ndof*i+1) = *(p+ndof*i+1);
		    *(force+ndof*i+2) = *(p+ndof*i+2);
		}
	    }

	    printf(" \n");
	    /*printf( "%f %f %f \n",*(force+7),*(P_global+7),*(dU+7));*/
	}

	if(dynamic_relax_flag || stress_update_flag)
	{

/* Set reaction forces */

	    if(dynamic_relax_flag)
	    {
		for( i = 0; i < numnp; ++i )
		{
		    if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0 )
		    {
			*(force+ndof*i) = *(P_global+ndof*i);
			*(force+ndof*i+1) = *(P_global+ndof*i+1);
			*(force+ndof*i+2) = *(P_global+ndof*i+2);
		    }
		}
	    }
	    else
	    {
		for( i = 0; i < numnp; ++i )
		{
		    if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0 )
		    {
			*(force+ndof*i) = *(P_global_CG+ndof*i);
			*(force+ndof*i+1) = *(P_global_CG+ndof*i+1);
			*(force+ndof*i+2) = *(P_global_CG+ndof*i+2);
		    }
		}
	    }

	    printf(" \n");
	    /*printf( "%f %f %f \n",*(force+7),*(P_global+7),*(dU+7));*/

	    for( k = 0; k < numel; ++k )
	    {
		matl_num = *(el_matl+k);
		Emod = matl[matl_num].E;
		Pois = matl[matl_num].nu;
		thickness = matl[matl_num].thick;

		K=Emod/(1.0-2*Pois)/3.0;
		G=Emod/(1.0+Pois)/2.0;
		Gt=thickness*Emod/(1.0+Pois)/2.0;

		for( j = 0; j < num_int; ++j )
		{

/* Count the number of times a particular node is part of an element */

		    node = *(connect+npel*k+j);

		    *(node_counter + node) += 1.0;

		    hydro = (stress[k].pt[j].xx + stress[k].pt[j].yy)*Pois/Emod;
		    strain[k].pt[j].xx = stress[k].pt[j].xx/(2.0*G) - hydro;
		    strain[k].pt[j].yy = stress[k].pt[j].yy/(2.0*G) - hydro;
		    strain[k].pt[j].xy = stress[k].pt[j].xy/G;

/* Calculate the principal strians */

		    xxaddyy = .5*(strain[k].pt[j].xx + strain[k].pt[j].yy);
		    xxsubyy = .5*(strain[k].pt[j].xx - strain[k].pt[j].yy);
		    xysq = strain[k].pt[j].xy*strain[k].pt[j].xy;

		    strain[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
			+ xysq);
		    strain[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
			+ xysq);

/* Add all the strains for a particular node from all the elements which share that node */

		    strain_node[node].xx += strain[k].pt[j].xx;
		    strain_node[node].yy += strain[k].pt[j].yy;
		    strain_node[node].xy += strain[k].pt[j].xy;
		    strain_node[node].I += strain[k].pt[j].I;
		    strain_node[node].II += strain[k].pt[j].II;

/* Calculate the principal stresses */

		    xxaddyy = .5*(stress[k].pt[j].xx + stress[k].pt[j].yy);
		    xxsubyy = .5*(stress[k].pt[j].xx - stress[k].pt[j].yy);
		    xysq = stress[k].pt[j].xy*stress[k].pt[j].xy;

		    stress[k].pt[j].I = xxaddyy + sqrt( xxsubyy*xxsubyy
			 + xysq);
		    stress[k].pt[j].II = xxaddyy - sqrt( xxsubyy*xxsubyy
			 + xysq);

/* Add all the stresses for a particular node from all the elements which share that node */

		    stress_node[node].xx += stress[k].pt[j].xx;
		    stress_node[node].yy += stress[k].pt[j].yy;
		    stress_node[node].xy += stress[k].pt[j].xy;
		    stress_node[node].I += stress[k].pt[j].I;
		    stress_node[node].II += stress[k].pt[j].II;
		}
	    }

/* Average all the stresses and strains at the nodes */

	    for( i = 0; i < numnp ; ++i )
	    {
		strain_node[i].xx /= *(node_counter + i);
		strain_node[i].yy /= *(node_counter + i);
		strain_node[i].xy /= *(node_counter + i);
		strain_node[i].I /= *(node_counter + i);
		strain_node[i].II /= *(node_counter + i);

		stress_node[i].xx /= *(node_counter + i);
		stress_node[i].yy /= *(node_counter + i);
		stress_node[i].xy /= *(node_counter + i);
		stress_node[i].I /= *(node_counter + i);
		stress_node[i].II /= *(node_counter + i);
	    }

	    for( k = 0; k < numel; ++k )
	    {
		surface_el_flag = 0;
		for( j = 0; j < num_int; ++j )
		{

/* Determine which elements are on the surface */

		   node = *(connect+npel*k+j);

		}
	    }
	}

	for( i = 0; i < numnp; ++i )
	{
	   if( *(id+ndof*i) > -1 )
		*(U+ndof*i) = *(coord+nsd*i)-*(coord0+nsd*i); 
	   if( *(id+ndof*i+1) > -1 )
		*(U+ndof*i+1) = *(coord+nsd*i+1)-*(coord0+nsd*i+1);
	   if( *(id+ndof*i+2) > -1 )
		*(U+ndof*i+2) = *(coord+nsd*i+2)-*(coord0+nsd*i+2);
	}

	check = qdwriter ( bc, connect, coord0, el_matl, force, id, matl,
		name, strain, strain_node, stress, stress_node, U);
	if(!check) printf( "Problems with qdwriter \n");

/*
	for( i = 0; i < numnp; ++i )
	{
		printf("\n  %3d   %14.5f  %14.5f  %14.5f",nsd*i,*(coord+nsd*i),
			*(P_global+nsd*i), *(force+nsd*i));
		printf("\n  %3d   %14.5f  %14.5f  %14.5f",nsd*i+1,*(coord+nsd*i+1),
			*(P_global+nsd*i+1), *(force+nsd*i+1));
		printf("\n  %3d   %14.5f  %14.5f  %14.5f",nsd*i+2,*(coord+nsd*i+2),
			*(P_global+nsd*i+2), *(force+nsd*i+2));
	}
	printf("\n");
*/

/* Write out the gnuplot file. */

	fdum = counter2*dt;

	check = GnuplotWriter2( gplot_dat_name, max, fdum );
	if(!check) printf( " Problems with GnuplotWriter2 \n");

	timec = clock();
	printf("elapsed CPU = %f\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(matl);
	free(mem_double);
	free(mem_double2);
	free(mem_int);
	free(mem_XYZI);

}

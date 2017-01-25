/*
    This program performs finite element analysis
    by reading in the data, doing assembling, and
    then solving the linear system for a truss

	Updated 12/4/06

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
#include <time.h>
#include "tsconst.h"
#include "tsstruct.h"

int tswriter ( BOUND , int *, double *, int *, double *, int *, MATL *,
        char *, STRAIN *, STRESS *, double *);

int solve(double *,double *,int *, int );

int decomp(double *, int *, int );

int tsKassemble(double *, int *, double *, int *, double *, int *, int *,
	int *, MATL *, STRAIN *, STRESS *, double *);

int diag( int *, int *, int, int, int, int);

int formlm( int *, int *, int *, int, int, int );

int tsformid( BOUND, int *);

int tsreader( BOUND , int *, double *, int *, double *, MATL *, 
	FILE *, STRESS *, double *);

int tsMemory( double **, int, int **, int, MATL **, int , XYZI **, int,
        STRAIN **, STRESS **, int );

int dof, sdof, analysis_flag, neqn, nmat, nmode, numel, numnp, sof;
int standard_flag, consistent_mass_flag, stress_read_flag, gauss_stress_flag;

int iteration_max, iteration_const, iteration;
double tolerance;

int main(int argc, char** argv)
{
	int i,j;
        int *id, *lm, *idiag, check;
        XYZI *mem_XYZI;
        int *mem_int, sofmi, sofmf, sofmSTRESS, sofmXYZI,
		ptr_inc;
        MATL *matl;
        double *mem_double;
        double fpointx, fpointy, fpointz;
	int *connect, *el_matl, dum;
	double *coord, *force, *U, *A;
	char name[30], buf[ BUFSIZ ];
        FILE *o1, *o2;
	BOUND bc;
        STRESS *stress;
        STRAIN *strain;
        long timec;
        long timef;
	double RAM_max, RAM_usable, RAM_needed, MEGS;

        sof = sizeof(double);

	memset(name,0,30*sizeof(char));

    	printf("What is the name of the file containing the \n");
    	printf("truss structural data? (example: tsp4_11.txt)\n");
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
	}

	fgets( buf, BUFSIZ, o1 );
        fscanf( o1, "%d %d %d %d\n ",&numel,&numnp,&nmat,&nmode);
        dof=numnp*ndof;
	sdof=numnp*nsd;

	standard_flag = 1;

/*   Begin allocation of meomory */

/* For the doubles */
        sofmf= sdof + 2*dof;

/* For the integers */
        sofmi= numel*npel + 2*dof + numel*npel*ndof + numel + numnp+1 + 1;

/* For the XYZI integers */
        sofmXYZI=numnp+1+1;

/* For the STRESS */
        sofmSTRESS=numel;

	check = tsMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZI, sofmXYZI, &strain, &stress, sofmSTRESS );
        if(!check) printf( " Problems with tsMemory \n");

/* For the doubles */
                                                ptr_inc = 0;
        coord=(mem_double+ptr_inc);             ptr_inc += sdof;
        force=(mem_double+ptr_inc);             ptr_inc += dof;
        U=(mem_double+ptr_inc);                 ptr_inc += dof;

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

        timec = clock();
        timef = 0;

	stress_read_flag = 1;
	check = tsreader( bc, connect, coord, el_matl, force, matl, o1,
		stress, U);
        if(!check) printf( " Problems with tsreader \n");

        check = tsformid( bc, id );
        if(!check) printf( " Problems with tsformid \n");

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

/*   allocate meomory for A, the skyline representation of the global stiffness */

        A=(double *)calloc((*(idiag+neqn-1)+1),sizeof(double));
	if(!A )
	{
		printf( "failed to allocate memory for A double\n ");
		exit(1);
	}

	analysis_flag = 1;
	memset(A,0,(*(idiag+neqn-1)+1)*sof);
        check = tsKassemble( A, connect, coord, el_matl, force, id, idiag, lm,
		matl, strain, stress, U);
        if(!check) printf( " Problems with tsKassembler \n");

#if DATA_ON
        printf( "\n\n This is the force matrix \n");
        for( i = 0; i < neqn; ++i )
        {
            printf( "\ndof %5d   %14.5f",i,*(force+i));
        }
        printf(" \n");
#endif

/* Perform LU Crout decompostion on the system */

        check = decomp(A,idiag,neqn);
        if(!check) printf( " Problems with decomp \n");

/* Solve the system */

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
        check = tsKassemble( A, connect, coord, el_matl, force, id, idiag, lm,
		matl, strain, stress, U);
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
#endif
	check = tswriter ( bc, connect, coord, el_matl, force, id, matl,
	        name, strain, stress, U);
	if(!check) printf( " Problems with tswriter \n");

	timec = clock();
        printf("\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	free(strain);
	free(stress);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZI);
}

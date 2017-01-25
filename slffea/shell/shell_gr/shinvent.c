/*
    This program converts a FEM shell mesh data file
    into an Open Inventor readable data file.

        Updated 6/10/00 

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../shell/shconst.h"
#include "../shell/shstruct.h"

int analysis_flag, sdof, dof, integ_flag, doubly_curved_flag, neqn, nmat, nmode,
	numel, numnp, sof;
int element_stress_read_flag, stress_read_flag, flag_quad_element;;

int shreader( BOUND , int *, double *, int *, double *, MATL *, char *, FILE *,
	STRESS *, SDIM *, double *);

int normcrossX(double *, double *, double *);

int crossX(double *coord, int tri[3],FILE *o3)
{
	int check;
	double coord_el[nsdsq],d1[nsd],d2[nsd],norm[nsd];

	*(coord_el)=*(coord+nsd*(*(tri)));
	*(coord_el+1)=*(coord+nsd*(*(tri))+1);
	*(coord_el+2)=*(coord+nsd*(*(tri))+2);
	*(coord_el+nsd*1)=*(coord+nsd*(*(tri+1)));
	*(coord_el+nsd*1+1)=*(coord+nsd*(*(tri+1))+1);
	*(coord_el+nsd*1+2)=*(coord+nsd*(*(tri+1))+2);
	*(coord_el+nsd*2)=*(coord+nsd*(*(tri+2)));
	*(coord_el+nsd*2+1)=*(coord+nsd*(*(tri+2))+1);
	*(coord_el+nsd*2+2)=*(coord+nsd*(*(tri+2))+2);

	*(d1)=*(coord_el)-*(coord_el+nsd*1);
	*(d1+1)=*(coord_el+1)-*(coord_el+nsd*1+1);
	*(d1+2)=*(coord_el+2)-*(coord_el+nsd*1+2);
	*(d2)=*(coord_el)-*(coord_el+nsd*2);
	*(d2+1)=*(coord_el+1)-*(coord_el+nsd*2+1);
	*(d2+2)=*(coord_el+2)-*(coord_el+nsd*2+2);

	check = normcrossX(d1, d2, norm);

	fprintf( o3,"%9.5f %9.5f %9.5f \n",*(norm),*(norm+1),*(norm+2));
	return 1;
}

int writer (int *connect, double *coord)
{
	FILE *o2,*o3,*o4;
	int i,tri[3],j,k,check;
	double fpointx, fpointy, fpointz,nvec[nsd];
	double coord_el[24];

	o2 = fopen( "shell.iv","w" );
	o3 = fopen( "norms","w" );

	fprintf(o2,"#Inventor V2.0 ascii\n\n");
	fprintf(o2,"Separator {\n    NormalBinding {\n    }\n  ");
	fprintf(o2,"    Rotationcoord {\n        axis    X\n        angle   0\n    }\n");
	fprintf(o2,"    Separator {\n        Label {\n");
	fprintf(o2,"            label       \"lpremol1\"\n        }\n");
	fprintf(o2,"        Material {\n");
	fprintf(o2,"            ambientColor        0.25 0.25 0.25\n");
	fprintf(o2,"            diffuseColor        0.666667 1 1\n        }\n");
	fprintf(o2,"        Separator {\n            Coordinate3 {\n                point   [");
	for( i = 0; i <= numnp-2; ++i )
	{
		fpointx=*(coord+nsd*i);
		fpointy=*(coord+nsd*i+1);
		fpointz=*(coord+nsd*i+2);
		fprintf(o2,"\n             %14.8f %14.8f %14.8f,",fpointx,fpointy,fpointz);
	}

	fpointx=*(coord+nsd*(numnp-1));
	fpointy=*(coord+nsd*(numnp-1)+1);
	fpointz=*(coord+nsd*(numnp-1)+2);
	fprintf(o2,"\n   %14.8f %14.8f %14.8f ]\n            }\n",fpointx,fpointy,fpointz);
	fprintf(o2,"            ShapeHints {\n");
	fprintf(o2,"                vertexOrdering  COUNTERCLOCKWISE\n");
	fprintf(o2,"                faceType        UNKNOWN_FACE_TYPE\n}\n");
	fprintf(o2,"            IndexedFaceSet {\n                coordIndex      [\n");

	for( j = 0; j < numel; ++j )
	{

		*(tri)=*(connect+npell4*j);
		*(tri+1)=*(connect+npell4*j+1);
		*(tri+2)=*(connect+npell4*j+2);
		check = crossX(coord, tri, o3);

		if( j == 0 )
		{
		   fprintf(o2, "                %d, %d, %d, ",
			*(tri),*(tri+1),*(tri+2));
		}
		else
		{
		   fprintf(o2, "                %d, %d, %d, %d, ",-1,
			*(tri),*(tri+1),*(tri+2));
		}

		*(tri)=*(connect+npell4*j);
		*(tri+1)=*(connect+npell4*j+2);
		*(tri+2)=*(connect+npell4*j+3);
		check = crossX(coord, tri, o3);
		if( j == numel -1 )
		{
		   fprintf(o2, "%d, %d, %d, %d ] \n            }\n",-1,
			*(tri),*(tri+1),*(tri+2));
		}
		else
		{
		   fprintf(o2, "%d, %d, %d, %d,  \n",-1,
			*(tri),*(tri+1),*(tri+2));
		}
	}
	fclose(o3);
	o4 = fopen( "norms","r" );
	fprintf(o2,"            Normal {\n                vector  [");
	for( j = 0; j < numel; ++j )
	{
		for( k = 0; k < 2; ++k )
		{
			   fscanf(o4, "%lf %lf %lf  \n",(nvec),(nvec+1),(nvec+2));
			   if( j == numel -1 && k == 1 )
			   {
				fprintf(o2, "                %lf %lf %lf  \n",
					*(nvec),*(nvec+1),*(nvec+2));
			   }
			   else
			   {
				fprintf(o2, "                %lf %lf %lf,  \n",
					*(nvec),*(nvec+1),*(nvec+2));
			   }
		}
	}
	fprintf(o2,"            ]\n            }\n");
	fprintf(o2,"            NormalBinding {\n");
	fprintf(o2,"                value   PER_FACE\n            }\n");
	fprintf(o2,"        }\n    }\n}\n");

	return 1;
}

int main(int argc, char** argv)
{
	int i, j;
	int *id, *lm, *idiag, check;
	XYZPhiI *mem_XYZPhiI;
	int *mem_int, sofmi, sofmf, sofmshf, sofmrotf, sofmXYZPhiI, sofmSDIM,
		ptr_inc;
	MATL *matl;
	double *mem_double;
	double fpointx, fpointy, fpointz, node_vec[nsd], fdum ;
	int *connect, *el_matl, dum;
	double *coord, *force, *U, *Uz_fib, *Voln, *A;
	char name[20], buf[ BUFSIZ ];
	FILE *o1;
	BOUND bc;
	STRESS *stress;
	STRAIN *strain;
	SDIM *stress_node, *mem_SDIM;
	double g;
	long timec;
	long timef;

	sof = sizeof(double);

	memset(name,0,20*sizeof(char));
	
	printf("What is the name of the file containing the \n");
	printf("shell structural data? \n");
	scanf( "%20s",name);

/*   o1 contains all the structural shell data  */

	o1 = fopen( name,"r" );

	if(o1 == NULL ) {
		printf("error on open\n");
		exit(1);
	}
	fgets( buf, BUFSIZ, o1 );
	fscanf( o1, "%d %d %d %d %d\n ",&numel,&numnp,&nmat,&nmode,&integ_flag);
	dof=numnp*ndof;

/*   Begin allocation of meomory */

/* For the doubles */
	sofmf=2*numnp*nsd+2*dof+numnp;
	mem_double=(double *)calloc(sofmf,sizeof(double));

	if(!mem_double )
	{
		printf( "failed to allocate memory for double\n ");
		exit(1);
	}
	                                    ptr_inc=0;
	coord=(mem_double+ptr_inc);         ptr_inc += 2*numnp*nsd;
	force=(mem_double+ptr_inc);         ptr_inc += dof;
	U=(mem_double+ptr_inc);             ptr_inc += dof;
	Uz_fib=(mem_double+ptr_inc);        ptr_inc += numnp;

/* For the materials */
	matl=(MATL *)calloc(nmat,sizeof(MATL));
	if(!matl )
	{
		printf( "failed to allocate memory for matl doubles\n ");
		exit(1);
	}

/* For the STRESS doubles */

	stress=(STRESS *)calloc(1,sizeof(STRESS));
	if(!stress )
	{
		printf( "failed to allocate memory for stress doubles\n ");
		exit(1);
	}

/* For the STRAIN doubles */

	strain=(STRAIN *)calloc(1,sizeof(STRAIN));
	if(!strain )
	{
		printf( "failed to allocate memory for strain doubles\n ");
		exit(1);
	}

/* For the integers */
	sofmi= numel*npell4+2*dof+numel*npell4*ndof+numel+numnp+1;
	mem_int=(int *)calloc(sofmi,sizeof(int));
	if(!mem_int )
	{
		printf( "failed to allocate memory for integers\n ");
		exit(1);
	}
	                                        ptr_inc = 0; 
	connect=(mem_int+ptr_inc);              ptr_inc += numel*npell4; 
	id=(mem_int+ptr_inc);                   ptr_inc += dof;
	idiag=(mem_int+ptr_inc);                ptr_inc += dof;
	lm=(mem_int+ptr_inc);                   ptr_inc += numel*npell4*ndof;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;

/* For the XYZPhiI integers */
	sofmXYZPhiI=numnp+1;
	mem_XYZPhiI=(XYZPhiI *)calloc(sofmXYZPhiI,sizeof(XYZPhiI));
	if(!mem_XYZPhiI )
	{
		printf( "failed to allocate memory for XYZPhiI integers\n ");
		exit(1);
	}
	                                      ptr_inc = 0; 
	bc.fix =(mem_XYZPhiI+ptr_inc);        ptr_inc += numnp;
	bc.num_fix=(mem_XYZPhiI+ptr_inc);     ptr_inc += 1;

/* For the SDIM doubles */
	sofmSDIM=1;
	mem_SDIM=(SDIM *)calloc(sofmSDIM,sizeof(SDIM));
	if(!mem_SDIM )
	{
		printf( "failed to allocate memory for SDIM integers\n ");
		exit(1);
	}
	                                        ptr_inc = 0;
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += 1;

	timec = clock();
	timef = 0;

	stress_read_flag = 0;
	element_stress_read_flag = 0;
	check = shreader( bc, connect, coord, el_matl, force, matl, name, o1,
		stress, stress_node, U);
	if(!check) printf( " Problems with reader \n");

	printf(" \n\n");

	check = writer( connect, coord );
	if(!check) printf( " Problems with writer \n");

	timec = clock();
	printf("\n elapsed CPU = %lf\n\n",( (double)timec)/800.);

	exit(1);

	free(mem_double);
	free(strain);
	free(stress);
	free(mem_int);
	free(mem_XYZPhiI);
	free(mem_SDIM);
}


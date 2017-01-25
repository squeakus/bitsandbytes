/*
    This program reads in the new input file and prepares it
    for graphical display.
  
   			Last Update 12/4/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../tri/trconst.h"
#include "../tri/trstruct.h"
#include "trstrcgr.h"
#include "../../common_gr/control.h"

/* glut header files */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/********************* These are all the subroutines **************************/

/******** Data management and calculations ********/

void force_vectors0(int , BOUND , double *, XYZF *);

void disp_vectors0(int , BOUND , double *);

void agvMakeAxesList(GLuint);

int trset( BOUND , int *, double *, XYZF *, SDIM *, ISDIM *,
	SDIM *, ISDIM *, double *, int *);

int parameter2( double *, SDIM *, SDIM *, double * );

int ReGetMemory2_gr2( XYZF **, int );

int trnormal_vectors (int *, double *, NORM * );

int reader_gr2( FILE *, SDIM *, SDIM *);

int trreader( BOUND, int *, double *, int *, double *, MATL *, char *,
	FILE *, SDIM *, SDIM *, double *);

int ReGetMemory_gr2( ISDIM **, ISDIM **, int, NORM **, int );

int ReGetMemory2( double **, int, int **, int, MATL **, int , XYZI **, int, 
	SDIM **, SDIM **, SDIM **, SDIM **, int, int );

int filecheck( char *, char *, FILE **, FILE **, FILE **, char *, int );

/******************************* GLOBAL VARIABLES **************************/

/****** FEA globals ******/
extern int dof, sdof, nmat, nmode, numel, numnp, plane_stress_flag;
extern int stress_read_flag, element_stress_read_flag, flag_3D;
extern XYZI *mem_XYZI;
extern XYZF *mem_XYZF;
extern int *mem_int;
extern double *mem_double;
extern NORM *mem_NORM;
extern double *coord, *coord0;
extern double *U;
extern int *connecter;
extern BOUND bc;
extern MATL *matl;
extern int *el_matl;
extern double *force;
extern SDIM *stress;
extern SDIM *strain;
extern SDIM *stress_node;
extern SDIM *strain_node;

/* Global variables for the mesh color and nodal data */

extern ISDIM *stress_color;
extern ISDIM *strain_color;
extern int *U_color, *el_matl_color;
extern NORM *norm, *norm0;
extern MATL *matl_crtl;

/* Global variables for drawing the axes */
extern GLuint AxesList, DispList, ForceList;   /* Display lists */

/* Global variables for drawing the force vectors */
extern XYZF *force_vec, *force_vec0;

/****** For drawing the Mesh Window ******/
extern double coord_rescale;

extern int input_flag, post_flag, matl_choice, node_choice, ele_choice, mode_choice;
extern int Before_flag, After_flag;
extern double amplify_factor;

int trGetNewMesh(void)
{
	int i, j, check;
	char *ccheck;
	int dum, dum1, dum2, dum3, dum4;
	double fpointx, fpointy, fpointz;
	int sofmi, sofmf, sofmSDIM, sofmSDIM_node, sofmISDIM,
		sofmXYZI, sofmXYZF, sofmNORM, ptr_inc;
	char name[30], name2[30], otr_exten[4], buf[ BUFSIZ ];
	int otr_exten_length = 4;
	FILE *o1, *o2, *o3;

/* Delete the old display lists */

	glDeleteLists(AxesList,1);
	if(input_flag)
	{
		glDeleteLists(DispList,1);
		glDeleteLists(ForceList,1);
	}

/* Initialize filenames */

	memset(name,0,30*sizeof(char));
	memset(name2,0,30*sizeof(char));
	memset(otr_exten,0,otr_exten_length*sizeof(char));

/* Initialize old variables */

	input_flag = 1;
	post_flag = 1;
	After_flag = 1;
	Before_flag = 0;
	mode_choice = 0;
	amplify_factor = 1.0;
	matl_choice = 0;
	node_choice = 0;
	ele_choice = 0;

	ccheck = strncpy(otr_exten,".otr",otr_exten_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	printf("What is the name of the input file containing the \n");
	printf("triangle structural data? (example: cir)\n");
	scanf( "%30s",name2);

/*   o1 contains all the structural data for input
     o3 contains all the structural data for postprocessing
     o2 is used to determine the existance of input and post files
*/
	o2 = fopen( name2,"r" );
	if(o2 == NULL ) {
		printf("Can't find file %30s\n", name2);
		exit(1);
	}
	/*printf( "%3d %30s\n ",name2_length,name2);*/

	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d %d %d %d %d\n ",&numel,&numnp,&nmat,&nmode,&plane_stress_flag);
	dof=numnp*ndof;
	sdof=numnp*nsd;
	nmode = abs(nmode);

/* Begin exmaining and checking for the existence of data files */

	check = filecheck( name, name2, &o1, &o2, &o3, otr_exten, otr_exten_length );
	if(!check) printf( " Problems with filecheck \n");

	if( input_flag )
	{
		fgets( buf, BUFSIZ, o1 );
		fscanf( o1, "%d %d %d %d %d\n ",&dum,&dum1,&dum2,&dum3,&dum4);
		printf( "%d %d %d %d %d\n ",dum,dum1,dum2,dum3,dum4);
		/*printf( "name %30s\n ",name);*/
	}
	if( post_flag )
	{
		fgets( buf, BUFSIZ, o3 );
		fscanf( o3, "%d %d %d %d %d\n ",&dum,&dum1,&dum2,&dum3,&dum4);
		printf( "%d %d %d %d %d\n ",dum,dum1,dum2,dum3,dum4);
		/*printf( "out %30s\n ",out);*/
	}

/*   Begin allocation of meomory */

/* For the doubles */
	sofmf=2*sdof+2*dof;

/* For the integers */
	sofmi= numel*npel+numel+numnp+1+1+dof;

/* For the XYZI integers */
	sofmXYZI=numnp+1+1;

/* For the SDIM */
	sofmSDIM=1;

/* For the SDIM_node */
	sofmSDIM_node=numnp;

/* For the ISDIM */
	sofmISDIM=numnp;

/* For the NORMS */
	sofmNORM=numel;
	if( input_flag && post_flag ) sofmNORM=2*numel;

	check = ReGetMemory2( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZI, sofmXYZI, &strain, &strain_node, &stress, &stress_node,
		sofmSDIM, sofmSDIM_node );
	if(!check) printf( " Problems with ReGetMemory2 \n");

	check = ReGetMemory_gr2( &strain_color, &stress_color, sofmISDIM, &mem_NORM,
		sofmNORM );
	if(!check) printf( " Problems with ReGetMemory_gr2 \n");

/* For the doubles */
	                                ptr_inc=0;
	coord=(mem_double+ptr_inc);     ptr_inc += sdof;
	coord0=(mem_double+ptr_inc);    ptr_inc += sdof;
	force=(mem_double+ptr_inc);     ptr_inc += dof;
	U=(mem_double+ptr_inc);         ptr_inc += dof;

/* For the materials */

	matl_crtl = matl;

/* For the integers */
	                                        ptr_inc = 0;
	connecter=(mem_int+ptr_inc);            ptr_inc += numel*npel;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;
	U_color=(mem_int+ptr_inc);              ptr_inc += dof;

	el_matl_color = el_matl;

/* For the XYZI integers */
	                                        ptr_inc = 0;
	bc.fix =(mem_XYZI+ptr_inc);             ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZI+ptr_inc);          ptr_inc += 1;

/* For the NORM doubles */
	                                        ptr_inc = 0;
	norm =(mem_NORM+ptr_inc);
	if( input_flag && post_flag )           ptr_inc += numel;
	norm0 =(mem_NORM+ptr_inc);              ptr_inc += numel;

/* If there is no post file, then set coord to coord0 */

	if( !post_flag )
	{
	    coord = coord0;
	    After_flag = 0;
	    Before_flag = 1;
	}

/* If there is no input file, then set coord0 to coord */

	if( !input_flag )
	{
	    /*coord0 = coord;*/
	    After_flag = 1;
	    Before_flag = 0;
	}

	stress_read_flag = 1;
	element_stress_read_flag = 0;
	if( post_flag )
	{
		check = trreader( bc, connecter, coord, el_matl, force, matl, name,
			o3, stress, stress_node, U);
		if(!check) printf( " Problems with trreader \n");
		stress_read_flag = 0;

/* In femtr.c, the line below is just the opposite where bc.num_fix[0].z = numnp for 2-D
   problems.  But I have to set it to zero here for 2-D problems because I don't want to
   draw prescribed displacements in z for 2-D problems.
*/
		if(!flag_3D) bc.num_fix[0].z = 0;

		check = reader_gr2( o3, strain_node, stress_node);
		if(!check) printf( " Problems with reader_gr2 \n");
	}
	if( input_flag )
	{
		check = trreader( bc, connecter, coord0, el_matl, force, matl, name,
			o1, stress, stress_node, U);
		if(!check) printf( " Problems with trreader \n");

/* See above relating to the line below.  */

		if(!flag_3D) bc.num_fix[0].z = 0;

	}

	if( post_flag )
	{
		check = trnormal_vectors(connecter, coord, norm );
		if(!check) printf( " Problems with trnormal_vectors \n");
	}

	if( input_flag )
	{
		check = trnormal_vectors(connecter, coord0, norm0 );
		if(!check) printf( " Problems with trnormal_vectors \n");
	}

/* For the XYZF doubles */
	sofmXYZF=2*bc.num_force[0];
/*
   This is allocated seperately from ReGetMemory_gr2 because we need to know the
   number of force vectors read from trreader and stored in bc.num_force[0].
*/

	check = ReGetMemory2_gr2( &mem_XYZF, sofmXYZF );
	if(!check) printf( " Problems with ReGetMemory2_gr2 \n");

	                                        ptr_inc = 0;
	force_vec =(mem_XYZF+ptr_inc);          ptr_inc += bc.num_force[0];
	force_vec0 =(mem_XYZF+ptr_inc);         ptr_inc += bc.num_force[0];

/* Search for extreme values */
 
/* In mesh viewer, search for extreme values of nodal points, displacements
   and stress and strains to obtain viewing parameters and make color
   assignments.  Also initialize variables */

	check = parameter2( coord, strain_node, stress_node, U );
	if(!check) printf( " Problems with parameter2 \n");

/* Rescale undeformed coordinates */

	if( coord_rescale > 1.01 || coord_rescale < .99 )
	{
	   if( input_flag && post_flag )
	   {
		for( i = 0; i < numnp; ++i )
		{
			*(coord0+nsd*i) /= coord_rescale;
			*(coord0+nsd*i+1) /= coord_rescale;
			*(coord0+nsd*i+2) /= coord_rescale;
		}
	   }
	}

	if( !input_flag )
	{
	    for ( i = 0; i < numnp; ++i)
	    {
		*(coord0 + nsd*i) = *(coord + nsd*i) - *(U + ndof*i);
		*(coord0 + nsd*i + 1) = *(coord + nsd*i + 1) - *(U + ndof*i + 1);
		*(coord0 + nsd*i + 2) = *(coord + nsd*i + 2) - *(U + ndof*i + 2);
	    }
	}

	check = trset( bc, connecter, force, force_vec0, strain_node,
		strain_color, stress_node, stress_color, U, U_color );
	if(!check) printf( " Problems with trset \n");

	AxesList = glGenLists(1);
	agvMakeAxesList(AxesList);

	if( input_flag )
	{

/* create display list for displacement and force grapics vectors
   on undeformed mesh*/

	    DispList = glGenLists(1);
	    disp_vectors0(DispList, bc, coord0);

	    for( i = 0; i < bc.num_force[0]; ++i)
	    {
		fpointx = *(coord0+nsd*bc.force[i]);
		fpointy = *(coord0+nsd*bc.force[i] + 1);
		fpointz = *(coord0+nsd*bc.force[i] + 2);
		force_vec[i].x = fpointx - force_vec0[i].x;
		force_vec[i].y = fpointy - force_vec0[i].y;
		force_vec[i].z = fpointz - force_vec0[i].z;
	    }
    
	    ForceList = glGenLists(1);
	    force_vectors0(ForceList, bc, coord0, force_vec);
    
	}

	if( post_flag )
	{
/* create force grapics vectors for deformed mesh*/

	    for( i = 0; i < bc.num_force[0]; ++i)
	    {
		fpointx = *(coord+nsd*bc.force[i]);
		fpointy = *(coord+nsd*bc.force[i] + 1);
		fpointz = *(coord+nsd*bc.force[i] + 2);
		force_vec[i].x = fpointx - force_vec0[i].x;
		force_vec[i].y = fpointy - force_vec0[i].y;
		force_vec[i].z = fpointz - force_vec0[i].z;
	    }
	}
	return 1;
}

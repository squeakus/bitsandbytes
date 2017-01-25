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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"
#include "../../common_gr/control.h"

/* glut header files */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/********************* These are all the subroutines **************************/

/******** Data management and calculations ********/

int bmrotate( double *, double *, double *);

void bmdist_load_vectors0(int , BOUND , int *, double *, XYZF_GR * );

void bmforce_vectors0(int , BOUND , double *, XYZPhiF *);

void bmdisp_vectors0(int , BOUND , double *);

void agvMakeAxesList(GLuint);

int bmset( BOUND , CURVATURE *, ICURVATURE *, double *, QYQZ *, int *,
	double * , XYZPhiF *, MOMENT *, IMOMENT *, STRAIN *, ISTRAIN *,
	STRESS *, ISTRESS *, double *, int * );

int bmparameter(double *, CURVATURE *, MOMENT *, STRAIN *, STRESS *, double * );

int bmReGetMemory2_gr( XYZPhiF **, int, XYZF_GR ** , int, QYQZ **, int );

int bmreader_gr( FILE *, CURVATURE *, STRAIN *);

int bmreader( double *, BOUND , int *, double *, double *, int *, int *, double *,
	MATL *, MOMENT *, FILE *, STRESS *, double *);

int bmReGetMemory_gr( ICURVATURE **, IMOMENT **, ISTRAIN **, ISTRESS **, int );

int bmReGetMemory( double **, int , int **, int , MATL **, int , XYZPhiI **, int ,
	CURVATURE **, MOMENT **, STRAIN **, STRESS **, int );

int filecheck( char *, char *, FILE **, FILE **, FILE **, char *, int );

/******************************* GLOBAL VARIABLES **************************/

/****** FEA globals ******/
extern int dof, sdof, nmat, nmode, numel, numnp;
extern int stress_read_flag, element_stress_read_flag;
extern XYZPhiI *mem_XYZPhiI;
extern XYZPhiF *mem_XYZPhiF;
extern int *mem_int;
extern double *mem_double;
extern double *coord, *coord0;
extern double *U;
extern int *connecter;
extern BOUND bc;
extern MATL *matl;
extern int *el_matl;
extern int *el_type;
extern double *force, *axis_z, *axis_z0;
extern double *dist_load;
extern MOMENT *moment;
extern STRESS *stress;
extern CURVATURE *curve;
extern STRAIN *strain;

/* Global variables for the mesh color and nodal data */

extern IMOMENT *moment_color;
extern ICURVATURE *curve_color;
extern ISTRESS *stress_color;
extern ISTRAIN *strain_color;
extern int *U_color, *el_matl_color;
extern MATL *matl_crtl;

/* Global variables for drawing the axes */
extern GLuint AxesList, DispList, ForceList, Dist_LoadList;   /* Display lists */

/* Global variables for drawing the force vectors */
extern XYZPhiF *force_vec, *force_vec0;
extern QYQZ *dist_load_vec0;
extern XYZF_GR *dist_load_vec;

/****** For drawing the Mesh Window ******/
extern double coord_rescale;

extern int input_flag, post_flag, matl_choice, node_choice, ele_choice, mode_choice;
extern int Before_flag, After_flag;
extern double amplify_factor;

int bmGetNewMesh(void)
{
	int i, j, k, node0, node1, check;
	char *ccheck;
	int dum, dum1, dum2, dum3, dum4;
	double fpointx, fpointy, fpointz;
	int sofmi, sofmf, sofmSTRESS, sofmISTRESS, sofmSTRAIN, sofmXYZPhiI,
		sofmXYZPhiF, sofmQYQZ, sofmXYZF_GR, ptr_inc;
	double vec_in[3], vec_out[3], coord_el[3*npel], coord0_el[3*npel];
	char name[30], name2[30], obm_exten[4], buf[ BUFSIZ ];
	int obm_exten_length = 4;
	FILE *o1, *o2, *o3;

/* Delete the old display lists */

	glDeleteLists(AxesList,1);
	if(input_flag)
	{
		glDeleteLists(DispList,1);
		glDeleteLists(ForceList,1);
		glDeleteLists(Dist_LoadList,1);
	}

/* Initialize filenames */

	memset(name,0,30*sizeof(char));
	memset(name2,0,30*sizeof(char));
	memset(obm_exten,0,obm_exten_length*sizeof(char));

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

	ccheck = strncpy(obm_exten,".obm",obm_exten_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	printf("What is the name of the input file containing the \n");
	printf("beam structural data? (example: bridge)\n");
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
	fscanf( o2, "%d %d %d %d\n ",&numel,&numnp,&nmat,&nmode);
	dof=numnp*ndof;
	sdof=numnp*nsd;
	nmode = abs(nmode);

/* Begin exmaining and checking for the existence of data files */

	check = filecheck( name, name2, &o1, &o2, &o3, obm_exten, obm_exten_length );
	if(!check) printf( " Problems with filecheck \n");

	if( input_flag )
	{
		fgets( buf, BUFSIZ, o1 );
		fscanf( o1, "%d %d %d %d\n ",&dum,&dum1,&dum2,&dum3);
		printf( "%d %d %d %d\n ",dum,dum1,dum2,dum3);
		/*printf( "name %30s\n ",name);*/
	}
	if( post_flag )
	{
		fgets( buf, BUFSIZ, o3 );
		fscanf( o3, "%d %d %d %d\n ",&dum,&dum1,&dum2,&dum3);
		printf( "%d %d %d %d\n ",dum,dum1,dum2,dum3);
		/*printf( "out %30s\n ",out);*/
	}

/*   Begin allocation of meomory */

/* For the doubles */
	sofmf=2*sdof+numnp*(nsd-1)+numnp+2*dof+2*numel*nsd;

/* For the integers */
	sofmi= numel*npel+2*numel+numnp+1+numel+1+2+dof;

/* For the XYZPhiI integers */
	sofmXYZPhiI=numnp+1+1;

/* For the STRESS */
	sofmSTRESS=numel;

/* For the ISTRESS */
	sofmISTRESS=numel;

	check = bmReGetMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZPhiI, sofmXYZPhiI, &curve, &moment, &strain, &stress,
		sofmSTRESS );
	if(!check) printf( " Problems with bmReGetMemory \n");

	check =  bmReGetMemory_gr( &curve_color, &moment_color, &strain_color,
		&stress_color, sofmISTRESS);
	if(!check) printf( " Problems with bmReGetMemory_gr \n");

/* For the doubles */
	                                   ptr_inc=0;
	coord=(mem_double+ptr_inc);        ptr_inc += sdof;
	coord0=(mem_double+ptr_inc);       ptr_inc += sdof;
	dist_load=(mem_double+ptr_inc);    ptr_inc += numnp*(nsd-1);
	force=(mem_double+ptr_inc);        ptr_inc += dof;
	U=(mem_double+ptr_inc);            ptr_inc += dof;
	axis_z=(mem_double+ptr_inc);       ptr_inc += numel*nsd;
	axis_z0=(mem_double+ptr_inc);      ptr_inc += numel*nsd;

/* For the materials */

	matl_crtl = matl;

/* For the integers */
	                                        ptr_inc = 0;
	connecter=(mem_int+ptr_inc);            ptr_inc += numel*npel;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	el_type=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.dist_load=(mem_int+ptr_inc);         ptr_inc += numel+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;
	bc.num_dist_load=(mem_int+ptr_inc);     ptr_inc += 1;
	U_color=(mem_int+ptr_inc);              ptr_inc += dof;

	el_matl_color = el_matl;

/* For the XYZPhiI integers */
	                                        ptr_inc = 0;
	bc.fix =(mem_XYZPhiI+ptr_inc);          ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZPhiI+ptr_inc);       ptr_inc += 1;

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
		check = bmreader( axis_z, bc, connecter, coord, dist_load, el_matl,
			el_type, force, matl, moment, o3, stress, U);
		if(!check) printf( " Problems with bmreader \n");
		stress_read_flag = 0;

		check = bmreader_gr( o3, curve, strain);
		if(!check) printf( " Problems with bmreader_gr \n");
	}
	if( input_flag )
	{
		check = bmreader( axis_z0, bc, connecter, coord0, dist_load, el_matl,
			el_type, force, matl, moment, o1, stress, U);
		if(!check) printf( " Problems with bmreader \n");

	}

/* For the XYZPhiF doubles */
	sofmXYZPhiF=2*bc.num_force[0];

/* For the XYZF_GR doubles */
	sofmXYZF_GR=bc.num_dist_load[0];

/* For the QYQZ doubles */
	sofmQYQZ=bc.num_dist_load[0];
/*
   This is allocated seperately from bmReGetMemory2_gr because we need to know the
   number of force vectors read from bmreader and stored in bc.num_force[0] and
   bc.num_dist_load[0].
*/

	check = bmReGetMemory2_gr( &mem_XYZPhiF, sofmXYZPhiF, &dist_load_vec, sofmXYZF_GR,
		&dist_load_vec0, sofmQYQZ);
	if(!check) printf( " Problems with bmReGetMemory2_gr \n");

/* For the XYZPhiF doubles */
	                                           ptr_inc = 0;
	force_vec =(mem_XYZPhiF+ptr_inc);          ptr_inc += bc.num_force[0];
	force_vec0 =(mem_XYZPhiF+ptr_inc);         ptr_inc += bc.num_force[0];

/* Search for extreme values */
 
/* In mesh viewer, search for extreme values of nodal points, displacements
   and stress and strains to obtain viewing parameters and make color
   assignments.  Also initialize variables */

	check = bmparameter( coord, curve, moment, strain, stress, U );
	if(!check) printf( " Problems with bmparameter \n");

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
		*(coord0 + nsd*i) = *(coord+nsd*i) - *(U+ndof*i);
		*(coord0 + nsd*i + 1) = *(coord+nsd*i + 1) - *(U+ndof*i+1);
		*(coord0 + nsd*i + 2) = *(coord+nsd*i + 2) - *(U+ndof*i+2);
	    }
	}

	check = bmset( bc, curve, curve_color, dist_load, dist_load_vec0,
		el_type, force , force_vec0, moment, moment_color, strain,
		strain_color, stress, stress_color, U, U_color );

	if(!check) printf( " Problems with bmset \n");

	AxesList = glGenLists(1);
	agvMakeAxesList(AxesList);

	if( input_flag )
	{

/* create display list for displacement, force, and distributed load
   grapics vectors on undeformed mesh*/

	    DispList = glGenLists(1);
	    bmdisp_vectors0(DispList, bc, coord0);

	    for( i = 0; i < bc.num_force[0]; ++i)
	    {
		fpointx = *(coord0+nsd*bc.force[i]);
		fpointy = *(coord0+nsd*bc.force[i] + 1);
		fpointz = *(coord0+nsd*bc.force[i] + 2);
		force_vec[i].x = fpointx - force_vec0[i].x;
		force_vec[i].y = fpointy - force_vec0[i].y;
		force_vec[i].z = fpointz - force_vec0[i].z;
		force_vec[i].phix = fpointx - force_vec0[i].phix;
		force_vec[i].phiy = fpointy - force_vec0[i].phiy;
		force_vec[i].phiz = fpointz - force_vec0[i].phiz;
	    }
    
	    ForceList = glGenLists(1);
	    bmforce_vectors0(ForceList, bc, coord0, force_vec);
    
/* create distributed load grapics vectors for deformed mesh*/
	    for( k = 0; k < bc.num_dist_load[0]; ++k)
	    {
		node0 = *(connecter+bc.dist_load[k]*npel);
		node1 = *(connecter+bc.dist_load[k]*npel+1);

		*(coord0_el)=*(coord0+nsd*node0);
		*(coord0_el+1)=*(coord0+nsd*node0+1);
		*(coord0_el+2)=*(coord0+nsd*node0+2);

		*(coord0_el+3)=*(coord0+nsd*node1);
		*(coord0_el+4)=*(coord0+nsd*node1+1);
		*(coord0_el+5)=*(coord0+nsd*node1+2);

		*(vec_in) =  0.0;
		*(vec_in+1) =  dist_load_vec0[k].qy;
		*(vec_in+2) =  dist_load_vec0[k].qz;

		check = bmrotate(coord0_el, vec_in, vec_out);
		if(!check) printf( " Problems with bmrotate \n");

		dist_load_vec[k].x = *(vec_out);
		dist_load_vec[k].y = *(vec_out+1);
		dist_load_vec[k].z = *(vec_out+2);

  /*printf(" dist_load %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		k, dist_load_vec0[k].qy, dist_load_vec0[k].qz,
		dist_load_vec[2*k].qy, dist_load_vec[2*k].qz,
		dist_load_vec[2*k+1].qy, dist_load_vec[2*k+1].qz);*/
	    }
	    Dist_LoadList = glGenLists(1);
	    bmdist_load_vectors0(Dist_LoadList, bc, connecter, coord0, dist_load_vec);
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
		    force_vec[i].phix = fpointx - force_vec0[i].phix;
		    force_vec[i].phiy = fpointy - force_vec0[i].phiy;
		    force_vec[i].phiz = fpointz - force_vec0[i].phiz;
	    }

/* create distributed load grapics vectors for deformed mesh*/

	    for( k = 0; k < bc.num_dist_load[0]; ++k)
	    {
		node0 = *(connecter+bc.dist_load[k]*npel);
		node1 = *(connecter+bc.dist_load[k]*npel+1);

		*(coord_el)=*(coord+nsd*node0);
		*(coord_el+1)=*(coord+nsd*node0+1);
		*(coord_el+2)=*(coord+nsd*node0+2);

		*(coord_el+3)=*(coord+nsd*node1);
		*(coord_el+4)=*(coord+nsd*node1+1);
		*(coord_el+5)=*(coord+nsd*node1+2);

		*(vec_in) =  0.0;
		*(vec_in+1) =  dist_load_vec0[k].qy;
		*(vec_in+2) =  dist_load_vec0[k].qz;

		check = bmrotate(coord_el, vec_in, vec_out);
		if(!check) printf( " Problems with bmrotate \n");

		dist_load_vec[k].x = *(vec_out);
		dist_load_vec[k].y = *(vec_out+1);
		dist_load_vec[k].z = *(vec_out+2);

  /*printf(" dist_load %d %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e\n",
		k, dist_load_vec0[k].qy, dist_load_vec0[k].qz,
		dist_load_vec[2*k].qy, dist_load_vec[2*k].qz,
		dist_load_vec[2*k+1].qy, dist_load_vec[2*k+1].qz);*/
	    }

	}

	return 1;
}

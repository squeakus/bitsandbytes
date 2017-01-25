/*
    This program shows the 3 dimensional model of the finite
    element mesh for shell elements.
  
	          Last Update 12/4/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "../shell/shconst.h"
#include "../shell/shstruct.h"
#include "shgui.h"
#include "shstrcgr.h"
#include "../../common_gr/control.h"
#include "../../common_gr/color_gr.h"

/* glut header files */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#if LINUX
/* X11 header files. */
#include <GL/glx.h>

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xmu/StdCmap.h>
#include <X11/keysym.h>

#include <X11/X.h>
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Shell.h>
#endif

/* The header files below can be used to incorporate
   widgets into your code */
#if 0
/* X11 Widget header files. */
#include <X11/Xaw/Command.h>
#include <X11/Xaw/Form.h>
#include <GL/xmesa.h>
#include <GL/MesaDrawingArea.h>

#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <GL/xmesa.h>
#include <GL/MesaMDrawingArea.h>

#define GLwMakeCurrent GLwMMakeCurrent
#endif

/********************* These are all the subroutines **************************/

/******** Data management and calculations ********/

void shforce_vectors0(int , BOUND , double *, XYZPhiF *);

void shdisp_vectors0(int , BOUND , double *);

void agvMakeAxesList(GLuint);

int shset( BOUND, int *, double *, XYZPhiF *, SDIM *,
	ISTRAIN *, SDIM *, ISTRESS *, double *U, int * );

int shparameter( double *, SDIM *, SDIM *, double * );

int shMemory2_gr( XYZPhiF **, int );

int dotX(double *,double *, double *, int );

int shlocal_vectors( double *, double *, double * );

int brnormal_vectors (int *, double *, NORM * );

int wenormal_vectors (int *, double *, NORM * );

int shTopCoordinates(int *, double *, int *, double *, MATL *, double *);

int node_normals( int *, double *, double *, double *, int, int);

int shreader_gr( FILE *, SDIM *, SDIM *);

int shreader( BOUND , int *, double *, int *, double *, MATL *,
	char *, FILE *, STRESS *, SDIM *, double *);

int shMemory_gr( ISTRAIN **, ISTRESS **, int, NORM **, int );

int shMemory( double **, int , int **, int , MATL **, int , XYZPhiI **,
	int , SDIM **, int, STRAIN **, STRESS **, int  );

int filecheck( char *, char *, FILE **, FILE **, FILE **, char *, int );

/******** For the Mesh ********/

void MeshKey_Special(int , int , int );

void shMeshKeys( unsigned char , int , int  );

void shmeshdraw(void);

void shrender(void);

void MeshReshape(int , int );

void shMeshDisplay(void);

void MeshInit(void);

/******** For the Control Panel ********/

void shControlMouse(int , int , int , int );

void ControlReshape(int , int );

void shControlDisplay(void);

void shMenu();

void ControlInit(void);

int ControlDimInit( void );

/******** For the Screen Dump ********/

void ScreenShot(int, int);

/* These are functions provided by Phillip Winston for
   movement of the mesh.*/

void agvHandleButton(int , int , int , int );

void agvHandleMotion(int , int );


/******************************* GLOBAL VARIABLES **************************/

/****** FEA globals ******/
int dof, sdof, nmat, nmode, numel, numnp, integ_flag, doubly_curved_flag;
int stress_read_flag, element_stress_read_flag, flag_quad_element;
XYZPhiI *mem_XYZPhiI;
XYZPhiF *mem_XYZPhiF;
int *mem_int;
double *mem_double;
SDIM *mem_SDIM;
NORM *mem_NORM;
double *coord, *coord0;
double *U, *Uz_fib, *fiber_xyz, *lamina_ref;
int *connecter;
BOUND bc;
MATL *matl;
int *el_matl;
double *force, *Voln;
STRESS *stress;
STRAIN *strain;
SDIM *stress_node;
SDIM *strain_node;

/* Global variables for the mesh color and nodal data */

ISTRESS *stress_color;
ISTRAIN *strain_color;
int *U_color, *el_matl_color;
NORM *norm, *norm0;
MATL *matl_crtl;

/* This used to be in MeshInit */

GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat mat_shininess[] = { 50.0 };
GLfloat light_position[] = { 1.0, 1.0, 6.0, 0.0 };

/****** graphics globals ******/

int choice_stress = 0;
int ControlWindow, MeshWindow;

/* Determines the step size for keyboard and Control panel movement */

double step_sizex = .1, step_sizey = .1, step_sizez = .1;

/****** Translation Variables ******/
double left_right, up_down, in_out, left_right0, up_down0, in_out0;

/****** Rotation Variables ******/
double xAngle = 0.0, yAngle = 0.0, zAngle = 0.0;

/****** Cross Section Plane Translation Variables ******/
double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
	cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;

/* Global variables for drawing the axes */
GLuint AxesList, DispList, ForceList;   /* Display lists */
double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
double AxisLength_x, AxisLength_y, AxisLength_z, AxisLength_max, AxisPoint_step;

/* Global variables for drawing the force vectors */
XYZPhiF *force_vec, *force_vec0;

/****** For drawing the Mesh Window ******/
double left, right, top, bottom, near, far, fscale, coord_rescale;
double ortho_left, ortho_right, ortho_top, ortho_bottom,
	ortho_left0, ortho_right0, ortho_top0, ortho_bottom0;
int mesh_width, mesh_height;
int com_mesh_width0 = mesh_width0;
int com_mesh_height0 = mesh_height0;

/* These Variables are for the Control Panel */

int sof = sizeof(double);
int sofi = sizeof(int);
int row_number = rowdim;
int ControlDiv_y[rowdim + 2], ControlDiv_x[rowdim + 2];
int control_width, control_height;
int com_control_width0 = control_width0;
int com_control_height0 =  control_height0;
int current_width, current_height;

double ratio = scaleFactor, ratio2 = 1.0;
double ratio_width = 1.0;
double ratio_height = 1.0;

int boxMove_x, boxMove_y, boxTextMove_x, textMove_x, textMove_y[rowdim+2];
int Color_flag[rowdim];

char RotateData[3][25] = { "    0.00", "    0.00", "    0.00" };
char MoveData[3][25] = { "    0.00", "    0.00", "    0.00" };
char AmplifyData[25] = { " 1.000e+00"};
char BoxData[2*boxnumber+2][25] = { "", "", "", "", "", "", "", "",
	"", "", "", "", "", "", "", "", "", "" };
char BoxText[25];

int del_height = 0;
int del_width = 0;

double matl_choicef = 0, node_choicef = 0, ele_choicef = 0;

int textDiv_xa = textDiv_xa0;
int textDiv_xb = textDiv_xb0;
int boxTextMove_x = boxTextMove_x0;

/* These Variables partition the stresses, strains, and displacements */

double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1],
	Uphi_x_div[boxnumber+1], Uphi_y_div[boxnumber+1];
SDIM stress_div[boxnumber+1], strain_div[boxnumber+1];
double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
double max_Uphi_x, min_Uphi_x, del_Uphi_x, max_Uphi_y, min_Uphi_y, del_Uphi_y,
	max_Ux, min_Ux, del_Ux, max_Uy, min_Uy, del_Uy,
	max_Uz, min_Uz, del_Uz, absolute_max_U, absolute_max_coord;

/* These are the flags */

int input_flag = 1,          /* Tells whether an input file exists or not */
    post_flag = 1,           /* Tells whether a post file exists or not   */
    color_choice = 1,        /* Set to desired color analysis(range 1 - 24) */
    choice = 0,              /* currently unused */
    matl_choice = -1,        /* Set to which material to view  */
    node_choice = -1,        /* Set to which node to view  */
    ele_choice = -1,         /* Set to which element to view  */
    mode_choice = 0;         /* Set to the desired modal post file */

int input_color_flag = 0;    /* Used with input_flag to determine how to draw input mesh */
int ortho_redraw_flag = 0;   /* calls MeshReshape(currently not used) */
int Solid_flag = 1,          /* Selects between wire frame or solid model */
    Perspective_flag = 1,    /* Selects between orthographic and perspecive views */
    Render_flag = 0,         /* Selects between rendering or analyses */
    AppliedDisp_flag = 0,    /* Turns applied on and off */
    AppliedForce_flag = 0,   /* Turns applied force on and off */
    Material_flag = 0,       /* Turns material on and off */
    Node_flag = 0,           /* Turns Node ID on and off */
    Element_flag = 0,        /* Turns Element ID on and off */
    Axes_flag = 0,           /* Turns Axes on and off  */
    Outline_flag = 1,        /* Turns Element Outline on and off  */
    Transparent_flag = 0,    /* Turns Element Transparency on and off  */
    CrossSection_flag = 0;   /* Turns CrossSection_flag on and off  */
int Before_flag = 0,         /* Turns before mesh on and off */
    After_flag = 1,          /* Turns after mesh on and off */
    Both_flag = 0,           /* Turns both before and after on and off */
    Amplify_flag = 0;        /* Turns Amplification on and off */

double amplify_factor = 1.0; /* Amplifies deformed mesh for better viewing */
double amplify_step = 0.1;   /* Value of amplification icrement */
double amplify_step0 = 0.1;  /* Value of initial calculated amplification icrement */

int stress_flag = 0,       /* Tells whether stress viewing is on or off */
    strain_flag = 0,       /* Tells whether strain viewing is on and off */
    stress_strain = 0,     /* Used with above 2 flags to determine how to draw Control Panel */
    disp_flag = 0,         /* Tells whether displacement viewing is on and off */
    angle_flag = 0;        /* Tells whether angle viewing is on and off */


int main(int argc, char** argv)
{
	int i, j, check;
	char *ccheck;
	int dum, dum1, dum2, dum3, dum4;
	double fpointx, fpointy, fpointz, node_vec[3], fdum;
	double *node_counter, *fiber_vec;
	int sofmi, sofmf, sofmSTRESS, sofmISTRESS, sofmSTRAIN,
		sofmXYZPhiI, sofmXYZPhiF, sofmSDIM, sofmNORM, ptr_inc;
	char name[30], name2[30], osh_exten[4], buf[ BUFSIZ ];
	int osh_exten_length = 4;
	FILE *o1, *o2, *o3;

	right=0;
	top=0;
	left=1000;
	bottom=1000;
	fscale=0;
	near=1.0;
	far=10.0;
	/*mesh_width=500;
	mesh_height=20;
	control_width=1000;
	control_height=1500;*/

/* Initialize filenames */

	memset(name,0,30*sizeof(char));
	memset(name2,0,30*sizeof(char));
	memset(osh_exten,0,osh_exten_length*sizeof(char));

	ccheck = strncpy(osh_exten,".osh",osh_exten_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	printf("What is the name of the input file containing the \n");
	printf("shell structural data? (example: bubble4node)\n");
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
	fscanf( o2, "%d %d %d %d %d\n ",&numel,&numnp,&nmat,&nmode,&integ_flag);
	dof=numnp*ndof;
	sdof=numnp*nsd;
	nmode = abs(nmode);

/* Begin exmaining and checking for the existence of data files */

	check = filecheck( name, name2, &o1, &o2, &o3, osh_exten, osh_exten_length );
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
	sofmf=4*sdof+2*dof+numnp+2*numnp+sdof+numnp*nsdsq+numnp*nsd;

/* For the integers */
	sofmi= numel*npell4+numel+numnp+1+1+dof;

/* For the XYZPhiI integers */
	sofmXYZPhiI=numnp+1+1;

/* For the SDIM doubles */
	sofmSDIM = 2*2*numnp;

/* For the STRESS */
	sofmSTRESS=1;

/* For the ISTRESS */
	sofmISTRESS=numel;

/* For the NORMS */
	sofmNORM=numel;
	if( input_flag && post_flag ) sofmNORM=2*numel;

	check = shMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat,
		&mem_XYZPhiI, sofmXYZPhiI, &mem_SDIM, sofmSDIM, &strain,
		&stress, sofmSTRESS );
	if(!check) printf( " Problems with shMemory \n");

	check = shMemory_gr( &strain_color, &stress_color, sofmISTRESS,
		&mem_NORM, sofmNORM );
	if(!check) printf( " Problems with shMemory_gr \n");

/* For the doubles */
	                                        ptr_inc=0;
	coord=(mem_double+ptr_inc);             ptr_inc += 2*sdof;
	coord0=(mem_double+ptr_inc);            ptr_inc += 2*sdof;
	force=(mem_double+ptr_inc);             ptr_inc += dof;
	U=(mem_double+ptr_inc);                 ptr_inc += dof;
	Uz_fib=(mem_double+ptr_inc);            ptr_inc += numnp;
	node_counter=(mem_double+ptr_inc);      ptr_inc += 2*numnp;
	fiber_vec=(mem_double+ptr_inc);         ptr_inc += sdof;
	fiber_xyz=(mem_double+ptr_inc);         ptr_inc += numnp*nsdsq;
	lamina_ref=(mem_double+ptr_inc);        ptr_inc += numnp*nsd;

/* For the materials */

	matl_crtl = matl;

/* For the integers */
	                                        ptr_inc = 0;
	connecter=(mem_int+ptr_inc);            ptr_inc += numel*npell4;
	el_matl=(mem_int+ptr_inc);              ptr_inc += numel;
	bc.force =(mem_int+ptr_inc);            ptr_inc += numnp+1;
	bc.num_force=(mem_int+ptr_inc);         ptr_inc += 1;
	U_color=(mem_int+ptr_inc);              ptr_inc += dof;

	el_matl_color = el_matl;

/* For the XYZPhiI integers */
	                                        ptr_inc = 0;
	bc.fix =(mem_XYZPhiI+ptr_inc);          ptr_inc += numnp+1;
	bc.num_fix=(mem_XYZPhiI+ptr_inc);       ptr_inc += 1;

/* For the SDIM doubles */
	                                        ptr_inc = 0;
	stress_node=(mem_SDIM+ptr_inc);         ptr_inc += 2*numnp;
	strain_node=(mem_SDIM+ptr_inc);         ptr_inc += 2*numnp;

/* For the NORM doubles */
	                                        ptr_inc = 0;
	norm =(mem_NORM+ptr_inc);
	if( input_flag && post_flag )           ptr_inc += numel;
	norm0 =(mem_NORM+ptr_inc);              ptr_inc += numel;

/* If there is no post file, then set coord, norm to coord0, norm0 */

	if( !post_flag )
	{
	    coord = coord0;
	    After_flag = 0;
	    Before_flag = 1;
	}

/* If there is no input file, then set coord0, norm0 to coord, and norm */

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
		check = shreader( bc, connecter, coord, el_matl, force, matl,
			name, o3, stress, stress_node, U);
		if(!check) printf( " Problems with shreader \n");
		stress_read_flag = 0;

		check = shreader_gr( o3, strain_node, stress_node);
		if(!check) printf( " Problems with shreader_gr \n");

		if(!doubly_curved_flag)
		{
/*  In the case of a singly curved shell, I use the function node_normals calculate
    vectors in the fiber direction for each node.
*/
		    check = node_normals( connecter, coord, fiber_vec,
			node_counter, numel, numnp);
		    if(!check) printf( " Problems with fiber_vec \n");

		    shTopCoordinates(connecter, coord, el_matl, fiber_vec, matl,
			node_counter);

		    memset(node_counter,0,numnp*sizeof(double));
		    memset(fiber_vec,0,sdof*sizeof(double));
		}

	}

	if( input_flag )
	{
		check = shreader( bc, connecter, coord0, el_matl, force, matl,
			name, o1, stress, stress_node, U);
		if(!check) printf( " Problems with shreader \n");

		if(!doubly_curved_flag)
		{
/*  In the case of a singly curved shell, I use the function node_normals calculate
    vectors in the fiber direction for each node.
*/
		    check = node_normals( connecter, coord0, fiber_vec,
			node_counter, numel, numnp);
		    if(!check) printf( " Problems with fiber_vec \n");

		    shTopCoordinates(connecter, coord0, el_matl, fiber_vec, matl,
			node_counter);

		    memset(node_counter,0,numnp*sizeof(double));
		    memset(fiber_vec,0,sdof*sizeof(double));
		}
	}

	if( post_flag )
	{
		if(flag_quad_element)
		{
		    check = brnormal_vectors(connecter, coord, norm );
		    if(!check) printf( " Problems with brnormal_vectors \n");
		}
		else
		{
		    check = wenormal_vectors(connecter, coord, norm );
		    if(!check) printf( " Problems with wenormal_vectors \n");
		}

		if( !input_flag )
		{
		   check = shlocal_vectors( coord, lamina_ref, fiber_xyz );
		   if(!check) printf( " Problems with shlocal_vectors\n");
		}
	}

	if( input_flag )
	{
		if(flag_quad_element)
		{
		    check = brnormal_vectors(connecter, coord0, norm0 );
		    if(!check) printf( " Problems with brnormal_vectors \n");
		}
		else
		{
		    check = wenormal_vectors(connecter, coord0, norm0 );
		    if(!check) printf( " Problems with wenormal_vectors \n");
		}

		check = shlocal_vectors( coord0, lamina_ref, fiber_xyz );
		if(!check) printf( " Problems with shlocal_vectors\n");
	}

/* For the XYZPhiF doubles */
	sofmXYZPhiF=2*bc.num_force[0];
/*
   This is allocated seperately from shMemory_gr because we need to know the
   number of force vectors read from shreader and stored in bc.num_force[0].
*/

	check = shMemory2_gr( &mem_XYZPhiF, sofmXYZPhiF );
	if(!check) printf( " Problems with shMemory2_gr \n");

	                                           ptr_inc = 0;
	force_vec =(mem_XYZPhiF+ptr_inc);          ptr_inc += bc.num_force[0];
	force_vec0 =(mem_XYZPhiF+ptr_inc);         ptr_inc += bc.num_force[0];

/* Search for extreme values */
 
/* In mesh viewer, search for extreme values of nodal points, displacements
   and stress and strains to obtain viewing parameters and make color
   assignments.  Also initialize variables */

	check = shparameter( coord, strain_node, stress_node, U );
	if(!check) printf( " Problems with shparameter \n");

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

			*(coord0+nsd*(i+numnp)) /= coord_rescale;
			*(coord0+nsd*(i+numnp) + 1) /= coord_rescale;
			*(coord0+nsd*(i+numnp) + 2) /= coord_rescale;
		}
	   }
	}

/* Calcuate the local z fiber displacement on each node */

	for( i = 0; i < numnp; ++i )
	{
/* Calcuate the projection of displacement vector to local z fiber vector */

	   *(node_vec) = *(coord+nsd*(numnp+i))-*(coord+nsd*i);
	   *(node_vec+1) = *(coord+nsd*(numnp+i)+1)-*(coord+nsd*i+1);
	   *(node_vec+2) = *(coord+nsd*(numnp+i)+2)-*(coord+nsd*i+2);
	   fdum = *(node_vec)*(*(node_vec)) +
		*(node_vec+1)*(*(node_vec+1)) + *(node_vec+2)*(*(node_vec+2));
	   fdum = sqrt(fdum);
	   *(node_vec) /= fdum;
	   *(node_vec+1) /= fdum;
	   *(node_vec+2) /= fdum;
	   check = check=dotX( (Uz_fib+i), (node_vec), (U+ndof*i), nsd);
	   if(!check) printf( " Problems with dotX \n");

	   /*printf("\n node %3d %14.9f %14.9f %14.9f %14.9f %14.9f",
		i,*(Uz_fib+i),fdum,*(node_vec), *(node_vec+1), *(node_vec+2));*/
	}

	if( !input_flag )
	{
/* Calcuate the local z fiber displacement on each node */

	    for( i = 0; i < numnp; ++i )
	    {
/* Calcuate the projection of displacement vector to local z fiber vector */

#if 0
		*(node_vec) = *(coord+nsd*(numnp+i))-*(coord+nsd*i);
		*(node_vec+1) = *(coord+nsd*(numnp+i)+1)-*(coord+nsd*i+1);
		*(node_vec+2) = *(coord+nsd*(numnp+i)+2)-*(coord+nsd*i+2);
		fdum = *(node_vec)*(*(node_vec)) +
		    *(node_vec+1)*(*(node_vec+1)) + *(node_vec+2)*(*(node_vec+2));
		fdum = sqrt(fdum);
		*(node_vec) /= fdum;
		*(node_vec+1) /= fdum;
		*(node_vec+2) /= fdum;
		check = check=dotX( (Uz_fib+i), (node_vec), (U+ndof*i), nsd);
		if(!check) printf( " Problems with dotX \n");
#endif

#if 0
		*(coord0 + nsd*i) =
			*(coord+nsd*i) - (*(U+ndof*i) +
			*(Uz_fib + i)*(*(U+ndof*i+4)));
		*(coord0 + nsd*i+1) =
			*(coord+nsd*i+1) - (*(U+ndof*i+1) -
			*(Uz_fib + i)*(*(U+ndof*i+3)));
		*(coord0 + nsd*i+2) =
			*(coord+nsd*i+2) - *(U+ndof*i+2);

		*(coord0 + nsd*(numnp+i)) =
			*(coord+nsd*(numnp+i)) - (*(U+ndof*i) +
			*(Uz_fib + i)*(*(U+ndof*i+4)));
		*(coord0 + nsd*(numnp+i) + 1) =
			*(coord+nsd*(numnp+i)+1) - (*(U+ndof*i+1) -
			*(Uz_fib + i)*(*(U+ndof*i+3)));
		*(coord0 + nsd*(numnp+i) + 2) =
			*(coord+nsd*(numnp+i)+2) - *(U+ndof*i+2);
#endif

/* Below is a calculation of the undeformed coordinates calculated as the difference between the
   deformed coordinates and the displacements.  It is based on equations 6.2.20-6.2.24 of "The Finite
   Element Method" by Thomas Hughes, page 388-389.  (David Benson, in "AMES 232 B Course Notes" page 50,
   does the same as Hughes.)  I have made some modifications, specifically, I replace za in equation
   6.2.23 with "Uz_fib" which is the projection of the displacement at a node in the fiber "e3"
   direction.

   It should be noted that in the "ANSYS User's Manual", page 12-12, these same equations mistakenly
   switch the fiber vectors e1 and e2(which are called "a" and "b" in ANSYS).  The first time these
   equations appear are in 12.5.1, 12.5.2, 12.5.3.  They also appear incorrectly like this later.
*/
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd)*(*(U+ndof*i+3)) +
			*(fiber_xyz + nsdsq*i)*(*(U+ndof*i+4));
		*(coord0 + nsd*i) =
			*(coord+nsd*i) - (*(U+ndof*i) +
			*(Uz_fib + i)*fdum);
		*(coord0 + nsd*(numnp+i)) =
			*(coord+nsd*(numnp+i)) - (*(U+ndof*i) +
			*(Uz_fib + i)*fdum);

		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 1)*(*(U+ndof*i+3)) +
			*(fiber_xyz + nsdsq*i + 1)*(*(U+ndof*i+4));
		*(coord0 + nsd*i+1) =
			*(coord+nsd*i+1) - (*(U+ndof*i+1) +
			*(Uz_fib + i)*fdum);
		*(coord0 + nsd*(numnp+i) + 1) =
			*(coord+nsd*(numnp+i)+1) - (*(U+ndof*i+1) +
			*(Uz_fib + i)*fdum);

		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 2)*(*(U+ndof*i+3)) +
			*(fiber_xyz + nsdsq*i + 2)*(*(U+ndof*i+4));
		*(coord0 + nsd*i+2) =
			*(coord+nsd*i+2) - (*(U+ndof*i+2) +
			*(Uz_fib + i)*fdum);
		*(coord0 + nsd*(numnp+i) + 2) =
			*(coord+nsd*(numnp+i)+2) - (*(U+ndof*i+2) +
			*(Uz_fib + i)*fdum);
	    }

	    /*for ( i = 0; i < numnp; ++i)
	    {
		*(coord0 + nsd*i) = *(coord+nsd*i) - *(U+ndof*i) -
			*(U+ndof*i+2)*(*(U+ndof*i+4));
		*(coord0 + nsd*i + 1) = *(coord+nsd*i+1) - *(U+ndof*i+1) +
			*(U+ndof*i+2)*(*(U+ndof*i+3));
		*(coord0 + nsd*i + 2) = *(coord+nsd*i+2) - *(U+ndof*i+2);
	    }*/
	}

	check = shset( bc, connecter, force, force_vec0, strain_node,
		strain_color, stress_node, stress_color, U, U_color );
	if(!check) printf( " Problems with shset \n");

/* Initialize the mesh viewer */


	glutInit(&argc, argv);
	glutInitWindowSize(mesh_width0, mesh_height0);
	glutInitWindowPosition(400, 215);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	MeshWindow = glutCreateWindow("SLFFEA");
	MeshInit ();

	AxesList = glGenLists(1);
	agvMakeAxesList(AxesList);

/* Below, I calculate force_vec[i].* for the force vectors graphics.  The reason I
   have coded things like this is because I think it gives me a slight improvement in
   speed.  When glutDisplayFunc displays the mesh, it continuously calls all the
   functions used in displaying the mesh like the subroutines which draw the force and
   prescribed displacement vectors.  This doesn't matter for the undeformed mesh where
   everything is drawn from display lists, but for the deformed mesh, it is an issue.
   So I calculate force_vec[i].* outside those functions, rather than simply passing
   force_vec0[i].* to the particular *force_vectors function and doing something like:

                fx = fpointx - force_vec0[node_num].x;
                fy = fpointy - force_vec0[node_num].y;
                fz = fpointz - force_vec0[node_num].z;

   There is probably only a small advantage, but that is the reason.
*/

	if( input_flag )
	{

/* create display list for displacement and force grapics vectors
   on undeformed mesh*/

	    DispList = glGenLists(1);
	    shdisp_vectors0(DispList, bc, coord0);

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
	    }
    
	    ForceList = glGenLists(1);
	    shforce_vectors0(ForceList, bc, coord0, force_vec);
    
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
	    }
	}

/* Initiate variables in Control Panel */

	memset(Color_flag,0,rowdim*sofi);

	check = ControlDimInit();
	if(!check) printf( " Problems with ControlDimInit \n");

/* call display function  */

	glutDisplayFunc(shMeshDisplay);

	glutReshapeFunc(MeshReshape);

/* Initialize Mouse Functions */

	glutMouseFunc(agvHandleButton);
	glutMotionFunc(agvHandleMotion);

/* Initialize Keyboard Functions */

	glutKeyboardFunc(shMeshKeys);
	glutSpecialFunc(MeshKey_Special);

/* Initialize the Control Panel */

	glutInitWindowSize(control_width0, control_height0);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB );
	ControlWindow = glutCreateWindow("SLFFEA Control Panel");

	ControlInit();
	shMenu();
	glutDisplayFunc(shControlDisplay);
	glutReshapeFunc(ControlReshape);

	glutMouseFunc(shControlMouse);

/* call function for hotkeys
 */
#if 0
	glutKeyboardFunc(ControlHandleKeys);
#endif
	glutMainLoop();

	free(strain);
	free(stress);
	free(mem_SDIM);
	free(strain_color);
	free(stress_color);
	free(mem_NORM);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZPhiI);
	free(mem_XYZPhiF);
	return 1;    /* ANSI C requires main to return int. */
}

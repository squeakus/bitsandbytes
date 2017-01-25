/*
    This program shows the 3 dimensional model of the finite
    element mesh for beam elements.
  
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
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmgui.h"
#include "bmstrcgr.h"
#include "../../common_gr/control.h"
#include "../../common_gr/color_gr.h"

/* glut header files */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#if LINUX
#include <GL/glx.h>

/* X11 header files. */
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

int bmrotate( double *, double *, double *);

void bmdist_load_vectors0(int , BOUND , int *, double *, XYZF_GR * );

void bmforce_vectors0(int , BOUND , double *, XYZPhiF *);

void bmdisp_vectors0(int , BOUND , double *);

void agvMakeAxesList(GLuint);

int bmset( BOUND , CURVATURE *, ICURVATURE *, double *, QYQZ *, int *, double * ,
	XYZPhiF *, MOMENT *, IMOMENT *, STRAIN *, ISTRAIN *, STRESS *, ISTRESS *,
	double *, int * );

int bmparameter(double *, CURVATURE *, MOMENT *, STRAIN *, STRESS *, double * );

int bmMemory2_gr( XYZPhiF **, int, XYZF_GR ** , int, QYQZ **, int );

int bmreader_gr( FILE *, CURVATURE *, STRAIN *);

int bmreader( double *, BOUND , int *, double *, double *, int *, int *, double *,
	MATL *, MOMENT *, FILE *, STRESS *, double *);

int bmMemory_gr( ICURVATURE **, IMOMENT **, ISTRAIN **, ISTRESS **, int );

int bmMemory( double **, int , int **, int , MATL **, int , XYZPhiI **, int ,
	CURVATURE **, MOMENT **, STRAIN **, STRESS **, int );

int filecheck( char *, char *, FILE **, FILE **, FILE **, char *, int );

/******** For the Mesh ********/

void bmMeshKey_Special(int , int , int );

void bmMeshKeys( unsigned char , int , int  );

void bmmeshdraw(void);

void bmrender(void);

void MeshReshape(int , int );

void bmMeshDisplay(void);

void MeshInit(void);

/******** For the Control Panel ********/

void bmControlMouse(int , int , int , int );

void ControlReshape(int , int );

void bmControlDisplay(void);

void bmMenu();

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
int dof, sdof, nmat, nmode, numel, numnp;
int stress_read_flag, element_stress_read_flag, stress_xyzx_flag;
XYZPhiI *mem_XYZPhiI;
XYZPhiF *mem_XYZPhiF;
int *mem_int;
double *mem_double;
double *coord, *coord0;
double *U;
int *connecter;
BOUND bc;
MATL *matl;
int *el_matl;
int *el_type;
double *force, *axis_z, *axis_z0;
double *dist_load;
MOMENT *moment;
STRESS *stress;
CURVATURE *curve;
STRAIN *strain;

/* Global variables for the mesh color and nodal data */

IMOMENT *moment_color;
ICURVATURE *curve_color;
ISTRESS *stress_color;
ISTRAIN *strain_color;
int *U_color, *el_matl_color;
MATL *matl_crtl;

/* This used to be in MeshInit */

GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat mat_shininess[] = { 50.0 };
GLfloat light_position[] = { 1.0, 1.0, 6.0, 0.0 };

/****** graphics globals ******/

int choice_stress_moment = 0;
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
GLuint AxesList, DispList, ForceList, Dist_LoadList;   /* Display lists */
double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
double AxisLength_x, AxisLength_y, AxisLength_z, AxisLength_max, AxisPoint_step;

/* Global variables for drawing the force vectors */
XYZPhiF *force_vec, *force_vec0;
QYQZ *dist_load_vec0;
XYZF_GR *dist_load_vec;

/****** For drawing the Mesh Window ******/
double left, right, top, bottom, near, far, fscale, coord_rescale;
double ortho_left, ortho_right, ortho_top, ortho_bottom,
	ortho_left0, ortho_right0, ortho_top0, ortho_bottom0;
int mesh_width, mesh_height;
int com_mesh_width0 = mesh_width0;
int com_mesh_height0 = mesh_height0;

/* These Variables are for the Control Panel */

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

double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1];
double Uphi_x_div[boxnumber+1], Uphi_y_div[boxnumber+1], Uphi_z_div[boxnumber+1];
SDIM stress_div[boxnumber+1], strain_div[boxnumber+1];
MDIM moment_div[boxnumber+1], curve_div[boxnumber+1];
double init_right, init_left, init_top,
	init_bottom, init_near, init_far, true_far, dim_max;
MDIM del_moment, del_curve, max_moment, min_moment,
	max_curve, min_curve;
SDIM del_stress, del_strain, max_stress, min_stress,
	max_strain, min_strain;
double max_Uphi_x, min_Uphi_x, del_Uphi_x, max_Uphi_y, min_Uphi_y, del_Uphi_y,
	max_Uphi_z, min_Uphi_z, del_Uphi_z, 
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
int Dist_Load_flag = 0,      /* Turns Distributed Load on and off  */
    Perspective_flag = 1,    /* Selects between orthographic and perspecive views */
    Render_flag = 0,         /* Selects between rendering or analyses */
    AppliedDisp_flag = 0,    /* Turns applied on and off */
    AppliedForce_flag = 0,   /* Turns applied force on and off */
    Material_flag = 0,       /* Turns material on and off */
    Node_flag = 0,           /* Turns Node ID on and off */
    Element_flag = 0,        /* Turns Element ID on and off */
    Axes_flag = 0,           /* Turns Axes on and off  */
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
	memset(obm_exten,0,obm_exten_length*sizeof(char));

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

	check = bmMemory( &mem_double, sofmf, &mem_int, sofmi, &matl, nmat, &mem_XYZPhiI,
		sofmXYZPhiI, &curve, &moment, &strain, &stress, sofmSTRESS );
	if(!check) printf( " Problems with bmMemory \n");

	check =  bmMemory_gr( &curve_color, &moment_color, &strain_color, &stress_color,
		sofmISTRESS);
	if(!check) printf( " Problems with bmMemory_gr \n");

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
   This is allocated seperately from bmMemory_gr because we need to know the
   number of force vectors read from bmreader and stored in bc.num_force[0]
   and bc.num_dist_load[0].
*/

	check = bmMemory2_gr( &mem_XYZPhiF, sofmXYZPhiF, &dist_load_vec, sofmXYZF_GR,
		&dist_load_vec0, sofmQYQZ);
	if(!check) printf( " Problems with bmMemory2_gr \n");

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

/* Initiate variables in Control Panel */

	memset(Color_flag,0,rowdim*sofi);

	check = ControlDimInit();
	if(!check) printf( " Problems with ControlDimInit \n");

/* call display function  */

	glutDisplayFunc(bmMeshDisplay);

	glutReshapeFunc(MeshReshape);

/* Initialize Mouse Functions */

	glutMouseFunc(agvHandleButton);
	glutMotionFunc(agvHandleMotion);

/* Initialize Keyboard Functions */

	glutKeyboardFunc(bmMeshKeys);
	glutSpecialFunc(bmMeshKey_Special);

/* Initialize the Control Panel */

	glutInitWindowSize(control_width0, control_height0);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB );
	ControlWindow = glutCreateWindow("SLFFEA Control Panel");

	ControlInit();
	bmMenu();
	glutDisplayFunc(bmControlDisplay);
	glutReshapeFunc(ControlReshape);

	glutMouseFunc(bmControlMouse);

/* call function for hotkeys
 */
#if 0
	glutKeyboardFunc(ControlHandleKeys);
#endif
	glutMainLoop();

	free(dist_load_vec);
	free(dist_load_vec0);
	free(curve);
	free(moment);
	free(strain);
	free(stress);
	free(curve_color);
	free(moment_color);
	free(strain_color);
	free(stress_color);
	free(matl);
	free(mem_double);
	free(mem_int);
	free(mem_XYZPhiI);
	free(mem_XYZPhiF);
	return 1;    /* ANSI C requires main to return int. */
}

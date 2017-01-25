/*
    This program contains the mesh key routine for the FEM GUI
    for truss elements.
  
                  Last Update 8/18/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"
#include "tsstrcgr.h"
#include "../../common_gr/control.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern int nmat, numnp, numel, dof;
extern double *coord, *coord0;
extern double *U;
extern int *connecter;
extern BOUND bc;
extern double *force;
extern SDIM *stress;
extern SDIM *strain;
extern XYZF *force_vec, *force_vec0;
extern ISDIM *stress_color;
extern ISDIM *strain_color;
extern int *U_color;

extern GLfloat MeshColor[boxnumber+5][4];
extern double step_sizex, step_sizey, step_sizez;
extern int choice_stress;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
	cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;
extern double ortho_left, ortho_right, ortho_top, ortho_bottom,
	ortho_left0, ortho_right0, ortho_top0, ortho_bottom0;
extern int ortho_redraw_flag;
extern double xAngle, yAngle, zAngle;
extern int mesh_width, mesh_height;
extern int input_flag, post_flag, color_choice,
    choice, matl_choice, node_choice, ele_choice;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z;
extern int input_color_flag;
extern int Perspective_flag, Render_flag,
    AppliedDisp_flag, AppliedForce_flag,
    Material_flag, Node_flag, Element_flag, Axes_flag,
    Transparent_flag, CrossSection_flag;
extern int Before_flag, After_flag,
    Both_flag, Amplify_flag;
extern double amplify_factor, amplify_step, amplify_step0;

void ScreenShot( int , int );

int tsset( BOUND , double *, XYZF *, SDIM *, ISDIM *, SDIM *,
	ISDIM *, double *, int *);

void tsReGetparameter(void);

int tsGetNewMesh(void);

void MeshReshape(int , int );

void tsMeshKeys( unsigned char key, int x, int y )
{

	int i, check;
	double fpointx, fpointy, fpointz;
/* Define Hotkeys */

/* 'i' zooms in on the mesh, 'o' zooms out */

	switch (key) {
	    case 'i':
		if ( Perspective_flag )
		{
			in_out += step_sizez;
		}
		else
		{
			ortho_left *= 0.90;
			ortho_right *= 0.90;
			ortho_bottom *= 0.90;
			ortho_top *= 0.90;
			MeshReshape( glutGet(GLUT_WINDOW_WIDTH),
				glutGet(GLUT_WINDOW_HEIGHT));
		}
		break;
	    case 'o':
		if ( Perspective_flag )
		{
			in_out -= step_sizez;
		}
		else
		{
			ortho_left *= 1.10;
			ortho_right *= 1.10;
			ortho_bottom *= 1.10;
			ortho_top *= 1.10;
			MeshReshape( glutGet(GLUT_WINDOW_WIDTH),
				glutGet(GLUT_WINDOW_HEIGHT));
		}
		break;

/* These keys control the selection of viewing stresses and strains and
   displacements. */

	    case '1':
		color_choice = 1;
		break;

	    case '!':
		color_choice = 10;
		break;


	    case '0':
		color_choice = 19;
		break;
	    case '-':
		color_choice = 20;
		break;
	    case '=':
		color_choice = 21;
		break;

/* 'n' selects the node to be viewed.  */

	    case 'n':
		color_choice = 31;
		input_color_flag = 0;
		AppliedForce_flag = 0;
		AppliedDisp_flag = 0;
		Element_flag = 0;
		Material_flag = 0;
		Node_flag = 1;

		printf("\n What is the desired node number?\n");
		scanf("%d", &node_choice);
		if ( node_choice > numnp - 1 )
		{
			node_choice = 0;
		}
		break;

/* 'e' selects the element to be viewed.  */

	    case 'e':
		color_choice = 32;
		input_color_flag = 0;
		AppliedForce_flag = 0;
		AppliedDisp_flag = 0;
		Element_flag = 1;
		Material_flag = 0;
		Node_flag = 0;

		printf("\n What is the desired element number?\n");
		scanf("%d", &ele_choice);
		if ( ele_choice > numel - 1 )
		{
			ele_choice = 0;
		}
		break;

/* 'm' selects the material to be viewed.  */

	    case 'm':
		color_choice = 30;
		input_color_flag = 0;
		AppliedForce_flag = 0;
		AppliedDisp_flag = 0;
		Element_flag = 0;
		Material_flag = 1;
		Node_flag = 0;

		printf("\n What is the desired material number?\n");
		scanf("%d", &matl_choice);
		if ( matl_choice > nmat - 1 )
		{
			matl_choice = 0;
		}
		break;

/* '>' and '<' amplify and shrink the displacements on the deformed object */

	    case '.':
		if( post_flag )
		{
			After_flag = 1;
			/*Amplify_flag = 1;*/
			amplify_step = amplify_step0;
			if( amplify_factor < 1.0 - SMALL2 )
				amplify_step = .1;
			amplify_factor += amplify_step;
/* Update Coordinates */
			for ( i = 0; i < numnp; ++i )
			{
			   *(coord + nsd*i) = *(coord0+nsd*i) +
				*(U+ndof*i)*amplify_factor;
			   *(coord + nsd*i+1) = *(coord0+nsd*i+1) +
				*(U+ndof*i+1)*amplify_factor;
			   *(coord + nsd*i+2) = *(coord0+nsd*i+2) +
				*(U+ndof*i+2)*amplify_factor;
			}

/* Update force graphics vectors */	
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
		break;
	    case ',':
		if( post_flag )
		{
			After_flag = 1;
			/*Amplify_flag = 1;*/
			amplify_step = amplify_step0;
			if( amplify_factor < 1.0 + amplify_step0 - SMALL2 )
			{
				amplify_step = .1;
			}
			amplify_factor -= amplify_step;
			if ( amplify_factor < 0.0 )
				amplify_factor = 0.0;
			/*printf("amplify factor %f \n", amplify_factor);*/
/* Update Coordinates */
			for ( i = 0; i < numnp; ++i )
			{
			   *(coord + nsd*i) = *(coord0+nsd*i) +
				*(U+ndof*i)*amplify_factor;
			   *(coord + nsd*i+1) = *(coord0+nsd*i+1) +
				*(U+ndof*i+1)*amplify_factor;
			   *(coord + nsd*i+2) = *(coord0+nsd*i+2) +
				*(U+ndof*i+2)*amplify_factor;
			}
	
/* Update force graphics vectors */	
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
		break;

/* The '<' and '>' keys move the Cross Section Plane on the Z
   Axes in the mesh window. */

	    case '<':
		cross_sec_in_out -= step_sizez;
		if( cross_sec_in_out < AxisMin_z )
			cross_sec_in_out = AxisMin_z;
		break;
	    case '>':
		cross_sec_in_out += step_sizez;
		if( cross_sec_in_out > AxisMax_z )
			cross_sec_in_out = AxisMax_z;
		break;

/* 'a' and 'b' turns on and off the deformed and undeformed mesh */

	    case 'a':
		After_flag = 1 - After_flag;
		break; 
	    case 'b':
		Before_flag = 1 - Before_flag;
		break; 

/* 'd' turns on and off the applied displacement vectors */

	    case 'd':
		AppliedDisp_flag = 1 - AppliedDisp_flag;
		break; 

/* Reset the rotation */

	    case 'c':
		xAngle = 0.0;
		yAngle = 0.0;
		zAngle = 0.0;
		break; 

/* Reset the translation */

	    case 'v':
		left_right = left_right0;
		up_down = up_down0;
		in_out = in_out0;
		if ( !Perspective_flag )
		{
			left_right = 0.0;
			up_down = 0.0;
			ortho_right = ortho_right0;
			ortho_left = ortho_left0;
			ortho_top = ortho_top0;
			ortho_bottom = ortho_bottom0;
			MeshReshape( glutGet(GLUT_WINDOW_WIDTH),
				glutGet(GLUT_WINDOW_HEIGHT));
		}
		cross_sec_left_right = cross_sec_left_right0;
		cross_sec_up_down = cross_sec_up_down0;
		cross_sec_in_out = cross_sec_in_out0;
		break; 

	    case 'f':
		AppliedForce_flag = 1 - AppliedForce_flag;
		break;
	    case 'g':
		check = tsGetNewMesh();
		if(!check) printf( " Problems with tsGetNewMesh\n");
		break;
	    case 'h':
		tsReGetparameter();
		check = tsset( bc, force, force_vec0, strain, strain_color, stress,
			stress_color, U, U_color);
		if(!check) printf( " Problems with tsparameter \n");
		if ( !Perspective_flag )
		{
			MeshReshape( glutGet(GLUT_WINDOW_WIDTH),
				glutGet(GLUT_WINDOW_HEIGHT));
		}
		break;
	    case 'p':
		Perspective_flag = 1 - Perspective_flag;
		MeshReshape( glutGet(GLUT_WINDOW_WIDTH),
			glutGet(GLUT_WINDOW_HEIGHT));
		left_right = left_right0;
		up_down = up_down0;
		/*in_out = in_out0;*/
		if ( !Perspective_flag )
		{
			left_right = 0.0;
			up_down = 0.0;
		}
		break;
	    case 'q':
		exit(1);
		break;
	    case 's':
		CrossSection_flag = 1 - CrossSection_flag;
		break;
	    case 't':
		Transparent_flag = 1 - Transparent_flag;
		break;
	    case 'x':
		Axes_flag = 1 - Axes_flag;
		break;
	    case 'y':
		ScreenShot( mesh_width, mesh_height );
		break;
	    case 27:
		exit(0);
		break;
	}

	input_color_flag = 0;
	if( color_choice < 10)
	     input_color_flag = 1;
	if( color_choice > 15 && color_choice < 19)
	     input_color_flag = 1;
	if( post_flag > 0 && color_choice < 30)
	     input_color_flag = 1;

	if(!post_flag) After_flag = 0;
	if(!input_flag) Before_flag = 0;

	glutPostRedisplay();
}


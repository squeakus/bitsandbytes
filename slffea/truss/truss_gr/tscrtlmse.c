/*
    This program contains the control mouse routine for the FEM GUI
    for truss elements.
  
                  Last Update 1/22/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../truss/tsconst.h"
#include "../truss/tsstruct.h"
#include "tsstrcgr.h"
#include "../../common_gr/control.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/****** FEA globals ******/

extern int dof, nmat, numnp, numel;
extern double *coord, *coord0;
extern double *U;
extern int *connecter;
extern BOUND bc;
extern XYZF *force_vec, *force_vec0;

/****** For drawing the Mesh Window ******/
extern double coord_rescale;

/* Global variables for the mesh color and nodal data */

extern int *el_matl_color;
extern MATL *matl_crtl;

/****** EXTERNAL VARIABLES ********/

extern int ControlDiv_y[rowdim + 2], ControlDiv_x[rowdim + 2];
extern int boxMove_x, boxMove_y, boxTextMove_x, textMove_x, textMove_y[rowdim+2];
extern int textDiv_xa, textDiv_xb;
extern double matl_choicef, node_choicef, ele_choicef;

extern int ControlWindow, MeshWindow;
extern double step_sizex, step_sizey, step_sizez;
extern int control_height, control_width, mesh_height, mesh_width;
extern double xAngle, yAngle, zAngle;
extern double amplify_factor, amplify_step, amplify_step0;
extern double left_right, up_down, in_out;
extern double left_right0, up_down0, in_out0;
extern double ortho_left, ortho_right, ortho_top, ortho_bottom,
	ortho_left0, ortho_right0, ortho_top0, ortho_bottom0;
extern int ortho_redraw_flag;

extern int input_flag, post_flag, color_choice, matl_choice, node_choice, ele_choice;
extern int input_color_flag;
extern int Dead_flag, Perspective_flag, Render_flag, AppliedDisp_flag,
	AppliedForce_flag, Material_flag, Node_flag, Element_flag, Axes_flag;
extern int Before_flag, After_flag, Both_flag, Amplify_flag; 
extern int stress_flag, strain_flag, stress_strain, disp_flag, angle_flag;

extern GLfloat yellow[4], orange[4], orangeRed[4], red[4], green[4],
	violetRed[4], magenta[4], purple[4], blue[4],
	white[4], grey[4], black[4];

extern char RotateData[3][25];
extern char MoveData[3][25];
extern char AmplifyData[25];
extern char BoxData[2*boxnumber+2][25];
extern char BoxText[25];

extern int Color_flag[rowdim];
extern double Ux_div[boxnumber+1], Uy_div[boxnumber+1], Uz_div[boxnumber+1];
extern SDIM stress_div[boxnumber+1];
extern SDIM strain_div[boxnumber+1];

void ScreenShot( int , int );

void tsControlMouse(int button, int state, int x, int y)
{
	int i, check, dum1, dum2;
	double fpointx, fpointy, fpointz;

	memset(BoxText,0,25*sizeof(char));
	for( i = 0; i < 2*boxnumber+2; ++i)
	{
		memset(BoxData[i],0,25*sizeof(char));
	}

	if (button == GLUT_LEFT_BUTTON)
	{
		if ( x < textDiv_xa )
		{

/* These are for the View Option Keys */

			if ( y >= ControlDiv_y[3] && y < ControlDiv_y[4] )
			{
/* Dead_flag Turned On */
				Dead_flag = 1;
			}
			if ( y >= ControlDiv_y[4] && y < ControlDiv_y[5] )
			{
/* Node ID Turned On increment up */
				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 1;
				node_choicef += .5;
				if ( node_choicef > 1.0 )
				{
					node_choicef = 0.0;
				}
				node_choice += (int)node_choicef;
				if ( node_choice > numnp - 1 )
				{
					node_choice = 0;
				}
				color_choice = 31;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[5] && y < ControlDiv_y[6] )
			{
/* Element ID Turned On increment up */
				/*ScreenShot( control_width, control_height );*/

				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 1;
				Material_flag = 0;
				Node_flag = 0;
				ele_choicef += .5;
				if ( ele_choicef > 1.0 )
				{
					ele_choicef = 0.0;
				}
				ele_choice += (int)ele_choicef;
				if ( ele_choice > numel - 1 )
				{
					ele_choice = 0;
				}
				color_choice = 32;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[6] && y < ControlDiv_y[7] )
			{
/* Material ID Turned On increment up */
				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 0;
				Node_flag = 0;
				Material_flag = 1;
				matl_choicef += .5;
				if ( matl_choicef > 1.0 )
				{
					matl_choicef = 0.0;
				}
				matl_choice += (int)matl_choicef;
				if ( matl_choice > nmat - 1 )
				{
					matl_choice = 0;
				}
				color_choice = 30;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[7] && y < ControlDiv_y[8] )
			{
/* Fixed Disp Turned On */
				/*AppliedForce_flag = 0;*/
				AppliedDisp_flag = 1;
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 0;
				/*if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}*/
			}
			if ( y >= ControlDiv_y[8] && y < ControlDiv_y[9] )
			{
/* Applied Force Turned On */
				AppliedForce_flag = 1;
				/*AppliedDisp_flag = 0;*/
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 0;
				/*if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}*/
			}
			if ( y >= ControlDiv_y[9] && y < ControlDiv_y[10] )
			{
/* Axes Turned On */
				Axes_flag = 1;
			}
/* These are for the Rotation Keys */
			if ( y >= ControlDiv_y[12] && y < ControlDiv_y[13] )
			{
/* Rotate -x */
				xAngle -= 5.0;
			}
			if ( y >= ControlDiv_y[13] && y < ControlDiv_y[14] )
			{
/* Rotate -y */
				yAngle -= 5.0;
			}
			if ( y >= ControlDiv_y[14] && y < ControlDiv_y[15] )
			{
/* Rotate -z */
				zAngle -= 5.0;
			}
			if ( y >= ControlDiv_y[15] && y < ControlDiv_y[16] )
			{
/* Reset Angles */
				xAngle = 0.0; yAngle = 0.0; zAngle = 0.0;
			}
/* These are for the Move Keys */
			if ( y >= ControlDiv_y[17] && y < ControlDiv_y[18] )
			{
/* Move -x */
				left_right -= step_sizex;
			}
			if ( y >= ControlDiv_y[18] && y < ControlDiv_y[19] )
			{
/* Move -y */
				up_down -= step_sizey;
			}
			if ( y >= ControlDiv_y[19] && y < ControlDiv_y[20] )
			{
/* Move -z */
				in_out -= step_sizez;
			}
			if ( y >= ControlDiv_y[20] && y < ControlDiv_y[21] )
			{
/* Reset Position */
				left_right = left_right0;
				up_down = up_down0;
				in_out = in_out0;

				ortho_right = ortho_right0;
				ortho_left = ortho_left0;
				ortho_top = ortho_top0;
				ortho_bottom = ortho_bottom0;
			}
/* These are for the Deformation Keys */
			if ( y >= ControlDiv_y[24] && y < ControlDiv_y[25] )
			{
/* Before Turned On */
				After_flag = 0;
				/*amplify_factor = 1.0;
				Amplify_flag = 0;*/
				Before_flag = 1;
				Both_flag = 0;
				/*disp_flag = 0;*/
				strain_flag = 0;
				if ( post_flag )
					stress_flag = 0;
			}
			if ( y >= ControlDiv_y[25] && y < ControlDiv_y[26] )
			{
/* After Turned On */
				After_flag = 1;
				Before_flag = 0;
				Both_flag = 0;
			}
			if ( y >= ControlDiv_y[26] && y < ControlDiv_y[27] )
			{
/* Both Before and After Turned On */
				After_flag = 1;
				/*AppliedForce_flag = 0;*/
				Both_flag = 1;
				Before_flag = 1;
				/*disp_flag = 0;
				AppliedDisp_flag = 0;*/
				/*Material_flag = 0;
				strain_flag = 0;
				stress_flag = 0;*/
			}
			if ( y >= ControlDiv_y[27] && y < ControlDiv_y[28] )
			{
/* Amplification increased */
			    if ( post_flag )
			    {
				After_flag = 1;
				amplify_step = amplify_step0;
				if( amplify_factor < 1.0 - SMALL2 )
					amplify_step = .1;
				amplify_factor += amplify_step;
				Amplify_flag = 1;
				/*AppliedForce_flag = 0;
				AppliedDisp_flag = 0;*/

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
			}

/* These are for the Engineering Analysis Option Keys */
			if ( y >= ControlDiv_y[30] )
			{
/* Stresses or displacement Turned On */
				if( post_flag)
				{
				   After_flag = 1;
				   Before_flag = 0;
				}
				Both_flag = 0;
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 0;
/* Stresses */

/* Stress XX*/
				strain_flag = 0;
				stress_flag = 1;
				color_choice = 10;
			}

/* Displacement */
			if ( y >= ControlDiv_y[32] )
			{
/* Displacement X*/
				angle_flag = 0;
				disp_flag = 1;
				stress_flag = 0;
				color_choice = 19;
			}
			if ( y >= ControlDiv_y[34] )
			{
/* Displacement Y*/
				angle_flag = 0;
				disp_flag = 1;
				stress_flag = 0;
				color_choice = 20;
			}
			if ( y >= ControlDiv_y[35] )
			{
/* Displacement Z*/
				angle_flag = 0;
				disp_flag = 1;
				stress_flag = 0;
				color_choice = 21;
			}
		}
		if ( x >= textDiv_xa && x < textDiv_xb )
		{
			if ( y >= ControlDiv_y[3] && y <= ControlDiv_y[4] )
			{
/* Dead_flag Turned Off */
				Dead_flag = 0;
			}
			if ( y >= ControlDiv_y[4] && y < ControlDiv_y[5] )
			{
/* Node ID Turned On increment down */
				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 1;
				node_choicef += .5;
				if ( node_choicef > 1.0 )
				{
					node_choicef = 0.0;
				}
				node_choice -= (int)node_choicef;
				if ( node_choice < 0 )
				{
					node_choice = numnp-1;
				}
				color_choice = 31;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[5] && y < ControlDiv_y[6] )
			{
/* Element ID Turned On increment down */
				/*ScreenShot( 350, 700);*/

				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 1;
				Material_flag = 0;
				Node_flag = 0;
				ele_choicef += .5;
				if ( ele_choicef > 1.0 )
				{
					ele_choicef = 0.0;
				}
				ele_choice -= (int)ele_choicef;
				if ( ele_choice < 0 )
				{
					ele_choice = numel-1;
				}
				color_choice = 32;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[6] && y < ControlDiv_y[7] )
			{
/* Material ID Turned On increment down */
				AppliedForce_flag = 0;
				angle_flag = 0;
				disp_flag = 0;
				AppliedDisp_flag = 0;
				Element_flag = 0;
				Node_flag = 0;
				Material_flag = 1;
				matl_choicef += .5;
				if ( matl_choicef > 1.0 )
				{
					matl_choicef = 0.0;
				}
				matl_choice -= (int)matl_choicef;
				if ( matl_choice < 0 )
				{
					matl_choice = nmat-1;
				}
				color_choice = 30;
				strain_flag = 0;
				stress_flag = 0;
				if ( Both_flag )
				{
					After_flag = 0;
					Before_flag = 1;
					Both_flag = 0;
				}
			}
			if ( y >= ControlDiv_y[7] && y <= ControlDiv_y[8] )
			{
/* Fixed Disp Turned Off */
				AppliedDisp_flag = 0;
			}
			if ( y >= ControlDiv_y[8] && y < ControlDiv_y[9] )
			{
/* Applied Force Turned Off */
				AppliedForce_flag = 0;
			}
			if ( y >= ControlDiv_y[9] && y < ControlDiv_y[10] )
			{
/* Axes Turned Off */
				Axes_flag = 0;
			}
/* These are for the Rotation Keys */
			if ( y >= ControlDiv_y[12] && y < ControlDiv_y[13] )
			{
/* Rotate +x */
				xAngle += 5.0;
			}
			if ( y >= ControlDiv_y[13] && y < ControlDiv_y[14] )
			{
/* Rotate +y */
				yAngle += 5.0;
			}
			if ( y >= ControlDiv_y[14] && y < ControlDiv_y[15] )
			{
/* Rotate +z */
				zAngle += 5.0;
			}
/* These are for the Move Keys */
			if ( y >= ControlDiv_y[17] && y < ControlDiv_y[18] )
			{
/* Move +x */
				left_right += step_sizex;
			}
			if ( y >= ControlDiv_y[18] && y < ControlDiv_y[19] )
			{
/* Move +y */
				up_down += step_sizey;
			}
			if ( y >= ControlDiv_y[19] && y < ControlDiv_y[20] )
			{
/* Move +z */
				in_out += step_sizez;
			}
/* These are for the Deformation Keys */
			if ( y >= ControlDiv_y[27] && y < ControlDiv_y[28] )
			{
/* Amplification decreased */
			    if( post_flag )
			    {
				After_flag = 1;
				amplify_step = amplify_step0;
				if( amplify_factor < 1.0 + amplify_step0 - SMALL2 )
					amplify_step = .1;
				amplify_factor -= amplify_step;
				/*Amplify_flag = 1;*/
				if ( amplify_factor < 0.0 )
				    amplify_factor = 0.0;

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
			}

/* These are for the Engineering Analysis Option Keys */
			if ( y >= ControlDiv_y[30] )
			{
/* Strains Turned On */
				if( post_flag)
				{
				   After_flag = 1;
				   Before_flag = 0;
				}
				Both_flag = 0;
				Element_flag = 0;
				Material_flag = 0;
				Node_flag = 0;
/* Strains */

/* Strain XX*/
				strain_flag = 1;
				color_choice = 1;
				stress_flag = 0;
			}
/* Displacement */
			if ( y >= ControlDiv_y[32] )
			{
/* Displacement X*/
				angle_flag = 1;
				disp_flag = 0;
				stress_flag = 0;
				color_choice = 19;
			}
			if ( y >= ControlDiv_y[34] )
			{
/* Displacement Y*/
				angle_flag = 2;
				disp_flag = 0;
				color_choice = 20;
			}
			if ( y >= ControlDiv_y[35] )
			{
/* Displacement Z*/
				angle_flag = 3;
				disp_flag = 0;
				color_choice = 21;
			}
		}
	}

	sprintf( RotateData[0], "%8.2f ", xAngle);
	sprintf( RotateData[1], "%8.2f ", yAngle);
	sprintf( RotateData[2], "%8.2f ", zAngle);

	sprintf( MoveData[0], "%8.2f ", left_right);
	sprintf( MoveData[1], "%8.2f ", up_down);
	sprintf( MoveData[2], "%8.2f ", in_out);

	sprintf( AmplifyData, "%10.3e ", amplify_factor);

	for(i = 29; i < rowdim ; ++i)
	{
		Color_flag[i] = 0;
	}

	switch (color_choice) {
		case 1:
			strncpy(BoxText, "strain XX", 9);
			Color_flag[29] = 1;
			strain_flag = 1;
			stress_flag = 0;
			sprintf( BoxData[0], "%10.3e ", strain_div[8].xx);
			sprintf( BoxData[2], "%10.3e ", strain_div[7].xx);
			sprintf( BoxData[4], "%10.3e ", strain_div[6].xx);
			sprintf( BoxData[6], "%10.3e ", strain_div[5].xx);
			sprintf( BoxData[8], "%10.3e ", strain_div[4].xx);
			sprintf( BoxData[10], "%10.3e ", strain_div[3].xx);
			sprintf( BoxData[12], "%10.3e ", strain_div[2].xx);
			sprintf( BoxData[14], "%10.3e ", strain_div[1].xx);
			sprintf( BoxData[16], "%10.3e ", strain_div[0].xx);
		break;
		case 10:
			strncpy(BoxText, "stress XX", 9);
			Color_flag[29] = 1;
			strain_flag = 0;
			stress_flag = 1;
			sprintf( BoxData[0], "%10.3e ", stress_div[8].xx);
			sprintf( BoxData[2], "%10.3e ", stress_div[7].xx);
			sprintf( BoxData[4], "%10.3e ", stress_div[6].xx);
			sprintf( BoxData[6], "%10.3e ", stress_div[5].xx);
			sprintf( BoxData[8], "%10.3e ", stress_div[4].xx);
			sprintf( BoxData[10], "%10.3e ", stress_div[3].xx);
			sprintf( BoxData[12], "%10.3e ", stress_div[2].xx);
			sprintf( BoxData[14], "%10.3e ", stress_div[1].xx);
			sprintf( BoxData[16], "%10.3e ", stress_div[0].xx);
		break;
		case 19:
			strncpy(BoxText, "disp X", 6);
			Color_flag[32] = 1;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 1;
			sprintf( BoxData[0], "%10.3e ", Ux_div[8]*coord_rescale);
			sprintf( BoxData[2], "%10.3e ", Ux_div[7]*coord_rescale);
			sprintf( BoxData[4], "%10.3e ", Ux_div[6]*coord_rescale);
			sprintf( BoxData[6], "%10.3e ", Ux_div[5]*coord_rescale);
			sprintf( BoxData[8], "%10.3e ", Ux_div[4]*coord_rescale);
			sprintf( BoxData[10], "%10.3e ", Ux_div[3]*coord_rescale);
			sprintf( BoxData[12], "%10.3e ", Ux_div[2]*coord_rescale);
			sprintf( BoxData[14], "%10.3e ", Ux_div[1]*coord_rescale);
			sprintf( BoxData[16], "%10.3e ", Ux_div[0]*coord_rescale);
		break;
		case 20:
			strncpy(BoxText, "disp Y", 6);
			Color_flag[33] = 1;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 1;
			sprintf( BoxData[0], "%10.3e ", Uy_div[8]*coord_rescale);
			sprintf( BoxData[2], "%10.3e ", Uy_div[7]*coord_rescale);
			sprintf( BoxData[4], "%10.3e ", Uy_div[6]*coord_rescale);
			sprintf( BoxData[6], "%10.3e ", Uy_div[5]*coord_rescale);
			sprintf( BoxData[8], "%10.3e ", Uy_div[4]*coord_rescale);
			sprintf( BoxData[10], "%10.3e ", Uy_div[3]*coord_rescale);
			sprintf( BoxData[12], "%10.3e ", Uy_div[2]*coord_rescale);
			sprintf( BoxData[14], "%10.3e ", Uy_div[1]*coord_rescale);
			sprintf( BoxData[16], "%10.3e ", Uy_div[0]*coord_rescale);
		break;
		case 21:
			strncpy(BoxText, "disp Z", 6);
			Color_flag[34] = 1;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 1;
			sprintf( BoxData[0], "%10.3e ", Uz_div[8]*coord_rescale);
			sprintf( BoxData[2], "%10.3e ", Uz_div[7]*coord_rescale);
			sprintf( BoxData[4], "%10.3e ", Uz_div[6]*coord_rescale);
			sprintf( BoxData[6], "%10.3e ", Uz_div[5]*coord_rescale);
			sprintf( BoxData[8], "%10.3e ", Uz_div[4]*coord_rescale);
			sprintf( BoxData[10], "%10.3e ", Uz_div[3]*coord_rescale);
			sprintf( BoxData[12], "%10.3e ", Uz_div[2]*coord_rescale);
			sprintf( BoxData[14], "%10.3e ", Uz_div[1]*coord_rescale);
			sprintf( BoxData[16], "%10.3e ", Uz_div[0]*coord_rescale);
		break;
		case 30:
			strncpy(BoxText, "Material", 8);
			Color_flag[5] = 1;
			input_color_flag = 0;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 0;
			Element_flag = 0;
			Node_flag = 0;
			Material_flag = 1;
			sprintf( BoxData[0], "%4d ", matl_choice);
			strncpy( BoxData[1], "Emod", 4);
			sprintf( BoxData[2], "%10.3e ", matl_crtl[matl_choice].E);
			strncpy( BoxData[3], "Mass", 4);
			sprintf( BoxData[4], "%10.3e ", matl_crtl[matl_choice].rho);
			strncpy( BoxData[5], "Area", 4);
			sprintf( BoxData[6], "%10.3e ", matl_crtl[matl_choice].area);
			sprintf( BoxData[8], " " );
			sprintf( BoxData[10], " " );
			sprintf( BoxData[12], " " );
			sprintf( BoxData[14], " " );
			sprintf( BoxData[16], " " );
		break;
		case 31:
			strncpy(BoxText, "Node", 4);
			Color_flag[3] = 1;
			input_color_flag = 0;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 0;
			Element_flag = 0;
			Material_flag = 0;
			Node_flag = 1;
			fpointx = *(coord + nsd*node_choice)*coord_rescale;
			fpointy = *(coord + nsd*node_choice + 1)*coord_rescale;
			fpointz = *(coord + nsd*node_choice + 2)*coord_rescale;
			if(!After_flag)
			{
				fpointx = *(coord0 + nsd*node_choice)*coord_rescale;
				fpointy = *(coord0 + nsd*node_choice + 1)*coord_rescale;
				fpointz = *(coord0 + nsd*node_choice + 2)*coord_rescale;
			}
			sprintf( BoxData[0], "%4d ", node_choice);
			strncpy( BoxData[2], "coord x", 7); 
			sprintf( BoxData[4], "%10.3e ", fpointx);
			strncpy( BoxData[6], "coord y", 7); 
			sprintf( BoxData[8], "%10.3e ", fpointy);
			strncpy( BoxData[10], "coord z", 7);
			sprintf( BoxData[12], "%10.3e ", fpointz);
			sprintf( BoxData[14], " " );
			sprintf( BoxData[16], " " );
		break;
		case 32:
			strncpy(BoxText, "Element", 7);
			Color_flag[4] = 1;
			input_color_flag = 0;
			strain_flag = 0;
			stress_flag = 0;
			angle_flag = 0;
			disp_flag = 0;
			Material_flag = 0;
			Node_flag = 0;
			Element_flag = 1;
			sprintf( BoxData[0], "%4d ", ele_choice);
			strncpy( BoxData[2], "Material", 8);
			sprintf( BoxData[4], "%4d ", *(el_matl_color+ele_choice));
			strncpy( BoxData[6], "Connect", 7);
			dum1 = *(connecter + npel*ele_choice);
			dum2 = *(connecter + npel*ele_choice+1);
			sprintf( BoxData[8], "%4d,%4d ",dum1, dum2);
			sprintf( BoxData[10], " " );
			sprintf( BoxData[12], " " );
			sprintf( BoxData[14], " " );
			sprintf( BoxData[16], " " );
		break;
	}

	input_color_flag = 0;

/* If there is a post file, then turn the input_color_flag on so that the before
   mesh will be drawn in pink.  If there is no post file, turn on the
   input_color_flag for every case except when stress/displacement analysis or material,
   element or node is selected.
 */

	if( color_choice < 10)
	     input_color_flag = 1;
	if( color_choice > 15 && color_choice < 19)
	     input_color_flag = 1;
	if( post_flag > 0 && color_choice < 30)
	     input_color_flag = 1;

	Color_flag[2] = 0;
/*
	Color_flag[3] = Perspective_flag;
	Color_flag[4] = Render_flag;
*/
	Color_flag[3] = Node_flag;
	Color_flag[4] = Element_flag;
	Color_flag[5] = Material_flag;
	Color_flag[6] = AppliedDisp_flag;
	Color_flag[7] = AppliedForce_flag;
	Color_flag[8] = Axes_flag;

	Color_flag[23] = Before_flag;
	Color_flag[24] = After_flag;
	Color_flag[25] = Both_flag;
	Color_flag[26] = Amplify_flag;

	if(!post_flag) After_flag = 0;
	if(!input_flag) Before_flag = 0;

	glutPostWindowRedisplay(ControlWindow);
	glutPostWindowRedisplay(MeshWindow);
}



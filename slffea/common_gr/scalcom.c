/*
    This program contains the scale reshape routine for every FEM GUI
    program.
  
                  Last Update 1/25/06

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
#include "control.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>


extern int Render_flag;
extern GLfloat MeshColor[boxnumber+5][4];
extern double near, ratio;
extern int Perspective_flag;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern double AxisLength_x, AxisLength_y, AxisLength_z,
	AxisLength_max;

/***** Scale Window Globals *****/

extern int ScaleDiv_y[], ScaleDiv_x[];
extern double scale_ratio, scale_ratio2;
/*extern int textDiv_xa, textDiv_xb;*/
extern double scale_ratio_width;
extern double scale_ratio_height;
extern double mesh_scale_ratio_width;
extern double mesh_scale_ratio_height;

/****** EXTERNAL VARIABLES ********/

extern int row_number;
extern int current_width;
extern int current_height;
extern int scale_height, scale_width;
extern int mesh_scale_height, mesh_scale_width;
extern int com_scale_height0, com_scale_width0;
extern int mesh_com_scale_height0, mesh_com_scale_width0;


extern int ScaleDiv_y[rowdim + 2], ScaleDiv_x[rowdim + 2];
extern char ControlText[][90];
extern int del_height;
extern int del_width;
extern double scale_ratio, scale_ratio2;
extern double mesh_scale_ratio, mesh_scale_ratio2;
extern int boxMove_x, boxMove_y, boxTextMove_x, scale_textMove_y[rowdim];
extern double mesh_boxMove_x, mesh_boxMove_y, mesh_boxMove_z, mesh_boxdim;
extern int scale_current_width, scale_current_height;
extern int scale_height, scale_width, mesh_height, mesh_width;
extern int mesh_scale_current_width, mesh_scale_current_height;
extern int mesh_scale_height, mesh_scale_width;
extern int stress_flag, strain_flag, stress_strain, disp_flag, thermal_flag;

extern GLfloat yellow[4], orangeRed[4], red[4], green[4],
	violetRed[4], magenta[4], purple[4], blue[4],
	white[4], grey[4], darkGrey[4], black[4];

extern GLfloat yellowRed[4], blueGreen[4], aqua[4], greenYellow[4];

extern char RotateData[3][10];
extern char MoveData[3][10];
extern char AmplifyData[10];
extern char BoxData[2*boxnumber+2][14];
extern char BoxText[10];

extern int Color_flag[rowdim];

void printText(const char *);

void printStrokedString(char *);

void ScaleDisplay(void)
{
	int i, j, dum = 0, dum2 = 0;
	GLfloat font_scale = 119.05 + 33.33;

	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glColor4fv(white);

	for( i = 0 ; i < (boxnumber+1)*2 ; ++i)
	{
		scale_textMove_y[i] = scale_current_height -
			2*textHeight - (int)(scale_ratio2*textHeightDiv*dum);
		++dum;
	}

/* Draw the Text */

/* Text lable for Color Scale Boxes */

	glColor4fv(white);
	boxTextMove_x = (int)(scale_ratio2*scale_boxTextMove_x0);
	/*for( i = 27 ; i < 28 ; ++i)*/
	for( i = 0 ; i < 1 ; ++i)
	{
		glLoadIdentity();
		glTranslatef (boxTextMove_x, scale_textMove_y[i], 0);
		glScalef( scale_ratio, scale_ratio, 1.0);
		printText( BoxText );
	}

/* Text for Color Scale Boxes */

	dum = 0;
	glColor4fv(white);
	/*for( i = 29 ; i < rowdim ; i += 2)*/
	for( i = 0 ; i < 2*(boxnumber+1) ; i += 1)
	{
		glLoadIdentity();
		glTranslatef (boxTextMove_x, scale_textMove_y[i] - 1.5*textHeight, 0);
		glScalef( scale_ratio, scale_ratio, 1.0);
		printText( BoxData[dum] );
		++dum;
	}

/* Begin Drawing the Color Scale Boxes */
	
	del_width = scale_current_width - scale_ratio2*scale_width0;
	del_height = scale_current_height - scale_ratio2*scale_height0;
	boxMove_x = (int)(scale_ratio2*(scale_width0 - left_indent));
	boxMove_y = del_height + (int)(scale_ratio2*bottom_indent);
	
	glLoadIdentity();
	glTranslatef (boxMove_x,boxMove_y,0);
	glScalef( scale_ratio2, scale_ratio2, 1.0);
	glColor4fv(MeshColor[0]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[1]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[2]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[3]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[4]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[5]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[6]);
	glRects(0,0,boxdim,boxdim);
	glTranslatef (0,boxHeight,0);
	glColor4fv(MeshColor[7]);
	glRects(0,0,boxdim,boxdim);

	glutSwapBuffers();

}

void ScaleReshape(int w, int h)
{
	int i, dum = 0;

	scale_width = w;
	scale_height = h;

/* Recalculate Parameters for Re-drawing of Scale Panel */

/* For the Text and the Color Scale Boxes */

	scale_ratio_width = (double) scale_width / com_scale_width0;
	scale_ratio_height = (double) scale_height / com_scale_height0;
	scale_ratio = scale_ratio_height;
	scale_current_height = scale_height;
	scale_current_width = scale_width;
	if( scale_ratio_width < scale_ratio_height)
	{
		scale_ratio = scale_ratio_width;
	}
	if( scale_ratio > 1.0)
	{
		scale_ratio = 1.0;
	}
	scale_ratio2 = scale_ratio;
	scale_ratio *= scaleFactor;

/* For the Mouse Parameters */

	for(i = 0; i < row_number + 2; ++i)
	{
		ScaleDiv_y[i] = (int)(scale_ratio2*textHeightDiv*dum - 5);
		++dum;
	}
/*
	textDiv_xa = (int)(scale_ratio2*textDiv_xa0);
	textDiv_xb = (int)(scale_ratio2*textDiv_xb0);
*/

	glViewport (0, 0, scale_width, scale_height);    /*  define the viewport */
	glMatrixMode (GL_PROJECTION);       /*  prepare for and then  */
	glLoadIdentity ();  /*  define the projection  */
	/*glOrtho (-2.0, 2.0, -2.0, 2.0,1, 40.0); */
	gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
	glMatrixMode (GL_MODELVIEW);       /*  back to modelview matrix    */
	glLoadIdentity();
}


/*
 *  This program contains the scale init routine for every FEM GUI
 *  program.
 *
 *                   Last Update 3/15/00
 */

void ScaleInit(void)
{
	glShadeModel (GL_FLAT);
	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
}

void MeshScaleDisplay(void)
{
	int i, j, dum = 0, dum2 = 0;
	GLfloat font_scale = 119.05 + 33.33;
	char chardum[20];
	double fdum, fdum2, fdum3;
	GLfloat d1[3], d2[3], norm_temp[] = {0.0, 0.0, 1.0};
	double coord_box[12], fpointx, fpointy, fpointz;

	glLoadIdentity();
	glColor4fv(white);

/* Begin Drawing the Color Scale Boxes */
	
	del_width = mesh_scale_current_width - scale_ratio2*mesh_width0;
	del_height = mesh_scale_current_height - scale_ratio2*mesh_height0;
	boxMove_x = (int)(mesh_scale_ratio2*(mesh_width0 - left_indent));
	boxMove_y = del_height + (int)(mesh_scale_ratio2*bottom_indent);

	glLoadIdentity();
	glScalef( fdum, fdum, 1.0);
	glTranslatef(mesh_boxMove_x, mesh_boxMove_y, mesh_boxMove_z);

	glLoadIdentity();
	fdum = mesh_boxdim;
	if(Perspective_flag) fdum = 0.5*mesh_boxdim;

	/*printf( "ffff %9.5f  %9.5f %9.5f %9.5f \n",
		mesh_boxdim, mesh_boxMove_x, mesh_boxMove_y, mesh_boxMove_z);*/

	fpointx = mesh_boxMove_x - 1.5*mesh_boxdim;
	fpointy = mesh_boxMove_y;
	fpointz = mesh_boxMove_z;

	fdum2 = 0.0;
	for( i = 0 ; i < boxnumber ; ++i)
	{
		coord_box[0] = fpointx;
		coord_box[1] = fpointy + fdum2*fdum;
		coord_box[2] = fpointz;

		coord_box[3] = fpointx + fdum;
		coord_box[4] = fpointy + fdum2*fdum;
		coord_box[5] = fpointz;

		coord_box[6] = fpointx + fdum;
		coord_box[7] = fpointy + fdum + fdum2*fdum;
		coord_box[8] = fpointz;

		coord_box[9] = fpointx;
		coord_box[10] = fpointy + fdum + fdum2*fdum;
		coord_box[11] = fpointz;

		glBegin(GL_QUADS);
		    if(!Render_flag) glColor4fv(MeshColor[i]);
		    if(Render_flag) glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[i]);
		    glNormal3fv(norm_temp);
		    glVertex3dv((coord_box));
		    glVertex3dv((coord_box + 3));
		    glVertex3dv((coord_box + 6));
		    glVertex3dv((coord_box + 9));
		glEnd();
		fdum2 += 1.0;
	}


	if(!Render_flag) glColor4fv(white);
	if(Render_flag) glMaterialfv(GL_FRONT, GL_DIFFUSE, white);
	glLineWidth (1.0);
	glLoadIdentity();

	fdum = .004*AxisLength_max;
	if(Perspective_flag) fdum = .002*AxisLength_max;

	fpointx += mesh_boxdim;
	fpointy += .5*mesh_boxdim*( -.25 + (double)boxnumber);
	if(!Perspective_flag) fpointy += .5*mesh_boxdim*( -.25 + (double)boxnumber);
	fdum2 = 0.0;
	for( i = 0 ; i < 2*(boxnumber+1) ; i += 1)
	{
		glLoadIdentity();
		glTranslatef(fpointx, fpointy - mesh_boxdim*fdum2, fpointz);
		glScalef( fdum, fdum, 1.0);
		printText( BoxData[i] );
		fdum2 += 0.25;
		if(!Perspective_flag) fdum2 += 0.25;
	}

	/*printf( "%9.5f %9.5f %9.5f %9.5f \n", boxMove_x, boxMove_y, mesh_scale_ratio2, ratio);*/

	glutSwapBuffers();
}


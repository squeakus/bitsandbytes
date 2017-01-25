/*
    This program draws the cutting planes which show the
    cross sections of the mesh.

			San Le

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

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z, IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern GLfloat wire_color[4], black[4], green[4], yellow[4],
	white[4], grey[4], darkGrey[4], black[4];
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
	cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;
extern double AxisLength_x, AxisLength_y, AxisLength_z,
	AxisLength_max, AxisPoint_step;
extern int Perspective_flag;

void CrossSetionPlaneDraw(void)
{
	double rot_vec[3];
	int check, i, j, k, dum;
	double fpointx, fpointy, fpointz;
	double fdum, fdum2, fdum4, textmove;

	fpointx = 0.95*AxisMax_x;
	fpointy = 0.95*AxisMax_y;
	fpointz = 0.95*AxisMax_z;

	glLineWidth (1.0);

/* Draw the X Axis Cross Section Plane */

	glBegin(GL_LINE_LOOP);
	  glColor4fv(white);
	  glVertex3f( cross_sec_left_right, AxisMin_y, AxisMin_z);
	  glVertex3f( cross_sec_left_right, fpointy, AxisMin_z);
	  glVertex3f( cross_sec_left_right, fpointy, fpointz);
	  glVertex3f( cross_sec_left_right, AxisMin_y, fpointz);
	glEnd();

/* Draw the Y Axis Cross Section Plane */

	glBegin(GL_LINE_LOOP);
	  glColor4fv(white);
	  glVertex3f( AxisMin_x, cross_sec_up_down, AxisMin_z);
	  glVertex3f( AxisMin_x, cross_sec_up_down, fpointz);
	  glVertex3f( fpointx, cross_sec_up_down, fpointz);
	  glVertex3f( fpointx, cross_sec_up_down, AxisMin_z);
	glEnd();

/* Draw the Z Axis Cross Section Plane */

	glBegin(GL_LINE_LOOP);
	  glColor4fv(white);
	  glVertex3f( AxisMin_x, AxisMin_y, cross_sec_in_out );
	  glVertex3f( fpointx, AxisMin_y, cross_sec_in_out );
	  glVertex3f( fpointx, fpointy, cross_sec_in_out );
	  glVertex3f( AxisMin_x, fpointy, cross_sec_in_out );
	glEnd();

}


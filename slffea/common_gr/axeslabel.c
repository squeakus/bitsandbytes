/*
    This program draws the lables of the axes as well as
    numbers the hash marks.

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
extern double left_right, up_down, in_out, left_right0,
	up_down0, in_out0, xAngle, yAngle, zAngle;
extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z,
	IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern double AxisLength_x, AxisLength_y, AxisLength_z,
	AxisLength_max, AxisPoint_step;

/****** For drawing the Mesh Window ******/
extern double coord_rescale;

extern int Perspective_flag;

int PointLocate( double *, double, double, double);


void AxesNumbers(void)
{
/*
    Draw the hash numerical label of the axes with the
    orientation of the labels always aligned with the
    viewers perspective.

		  Last Update 1/25/06
*/
	double point_vec[3];
	int check, i, j, k, dum;
	double fpointx, fpointy, fpointz;
	double fdum, fdum2, fdum4, textmove;
	char chardum[20];

/* For the X axis hash numerical label */

	glLineWidth (1.0);
	fdum4 = .004*AxisLength_max;

	dum = 1 + (int) (AxisLength_x/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_x;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = fdum;
		point_vec[1] = 0.0;
		point_vec[2] = 0.0;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}


/* For the Y axis hash numerical label */

	dum = 1 + (int) (AxisLength_y/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_y;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = 0.0;
		point_vec[1] = fdum;
		point_vec[2] = 0.0;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}


/* For the Z axis hash numerical label */

	dum = 1 + (int) (AxisLength_z/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_z;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = 0.0;
		point_vec[1] = 0.0;
		point_vec[2] = fdum;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}

	/*return 1;*/
}

void AxesNumbers2(void)
{
/*
    Draw the hash numerical label of the axes with the
    orientation of the labels aligned with its associate
    axis.

		  Last Update 1/25/06
*/
	double point_vec[3];
	int check, i, j, k, dum;
	double fpointx, fpointy, fpointz;
	double fdum, fdum2, fdum4, textmove;
	char chardum[20];

/* For the X axis hash numerical label */

	glLineWidth (1.0);
	fdum4 = .004*AxisLength_max;

	dum = 1 + (int) (AxisLength_x/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_x;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = fdum;
		point_vec[1] = 0.0;
		point_vec[2] = 0.0;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glRotatef (xAngle, 1, 0, 0);
		glRotatef (yAngle, 0, 1, 0);
		glRotatef (zAngle, 0, 0, 1);
		glRotatef ( -90, 0, 1, 0);
		glRotatef ( -90, 1, 0, 0);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}


/* For the Y axis hash numerical label */

	dum = 1 + (int) (AxisLength_y/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_y;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = 0.0;
		point_vec[1] = fdum;
		point_vec[2] = 0.0;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glRotatef (xAngle, 1, 0, 0);
		glRotatef (yAngle, 0, 1, 0);
		glRotatef (zAngle, 0, 0, 1);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}


/* For the Z axis hash numerical label */

	dum = 1 + (int) (AxisLength_z/AxisPoint_step);
	glColor4fv(white);
	fdum = (double)IAxisMin_z;
	for( i = 0; i < dum; ++i)
	{
		point_vec[0] = 0.0;
		point_vec[1] = 0.0;
		point_vec[2] = fdum;

		check = PointLocate(point_vec, xAngle, yAngle, zAngle);
		if(!check) printf( " Problems with PointLocate\n");

		fpointx = point_vec[0] + left_right;
		fpointy = point_vec[1] + up_down;
		fpointz = point_vec[2] + in_out;

		glLoadIdentity();
		glTranslatef( fpointx, fpointy, fpointz);
		glRotatef (xAngle, 1, 0, 0);
		glRotatef (yAngle, 0, 1, 0);
		glRotatef (zAngle, 0, 0, 1);
		glRotatef (  90, 1, 0, 0);
		glRotatef (  180, 0, 1, 0);
		glScalef( fdum4, fdum4, 1.0);
		sprintf( chardum, "%10.3e ", fdum*coord_rescale);
		printText( chardum );

		fdum += AxisPoint_step;
	}

	/*return 1;*/
}


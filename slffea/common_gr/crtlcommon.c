/*
    This source file contains all the control panel initialization functions
    for every FEM GUI program.
  
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

/***** Control Panel Globals *****/

extern int ControlDiv_y[], ControlDiv_x[];
extern double ratio, ratio2;
extern int textDiv_xa, textDiv_xb;
extern double ratio_width;
extern double ratio_height;

/****** EXTERNAL VARIABLES ********/

extern int row_number;
extern int current_width;
extern int current_height;
extern int control_height, control_width;
extern int com_control_height0, com_control_width0;

void ControlReshape(int w, int h)
{
/*
    This program contains the control reshape routine for every FEM GUI
    program.
  
	                Last Update 5/14/00
*/
	int i, dum = 0;

	control_width = w;
	control_height = h;

/* Recalculate Parameters for Re-drawing of Control Panel */

/* For the Text and the Color Scale Boxes */

	ratio_width = (double) control_width / com_control_width0;
	ratio_height = (double) control_height / com_control_height0;
	ratio = ratio_height;
	current_height = control_height;
	current_width = control_width;
	if( ratio_width < ratio_height)
	{
		ratio = ratio_width;
	}
	if( ratio > 1.0)
	{
		ratio = 1.0;
	}
	ratio2 = ratio;
	ratio *= scaleFactor;

/* For the Mouse Parameters */

	for(i = 0; i < row_number + 2; ++i)
	{
		ControlDiv_y[i] = (int)(ratio2*textHeightDiv*dum - 5);
		++dum;
	}
	textDiv_xa = (int)(ratio2*textDiv_xa0);
	textDiv_xb = (int)(ratio2*textDiv_xb0);

	glViewport (0, 0, control_width, control_height);    /*  define the viewport */
	glMatrixMode (GL_PROJECTION);       /*  prepare for and then  */
	glLoadIdentity ();  /*  define the projection  */
	/*glOrtho (-2.0, 2.0, -2.0, 2.0,1, 40.0); */
	gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
	glMatrixMode (GL_MODELVIEW);       /*  back to modelview matrix    */
	glLoadIdentity();
}


/*
 *  This program contains the control init routine for every FEM GUI
 *  program.
 *
 *                    Last Update 4/13/99
 */

void ControlInit(void)
{
	glShadeModel (GL_SMOOTH);
	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
}

/*
 *  This program initializes the value of the vertical spacing between
 *  the text on the Control Panel for every FEM GUI program.
 *
 *                    Last Update 4/14/99
 */

int ControlDimInit( void )
{
	int i, dum = 0;
	for(i = 0; i < row_number + 2; ++i)
	{
		ControlDiv_y[i] = textHeightDiv*dum - 5;
		++dum;
		/*printf ( "ControlDiv_y %3d %3d\n", i, ControlDiv_y[i]);*/
	}
	return 1;
}




/*
    This source file contains the special keys for the beam GUI as well as some
    additional keys.  Because I think you are only allowed one special key
    function, I have replaced "MeshKey_Special" in:

       ~/slffea-1.4/common_gr/mshcommon.c

    with this code.

	                Last Update 9/24/06

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

extern int color_choice;

extern double step_sizex, step_sizey, step_sizez;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double AxisMax_x, AxisMax_y, AxisMax_z, AxisMin_x, AxisMin_y, AxisMin_z;
extern double left, right, top, bottom, near, far, fscale;
extern int CrossSection_flag;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
        cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;

void bmMeshKey_Special(int key, int x, int y)
{

/* I needed these keys because the number keys are already taken by the
   stresses and strains.  So I am using function keys and some others for
   the curvature and moments.  Look in:

      /usr/include/GL/glut.h

   for a list of the special keys.
*/ 

  if(CrossSection_flag)
  {
/* The arrow keys move the Cross Section Plane on the X and Y
   Axes in the mesh window. */

	switch (key) {
	    case GLUT_KEY_UP:
		cross_sec_up_down += step_sizey;
		if( cross_sec_up_down > AxisMax_y )
			cross_sec_up_down = AxisMax_y;
		break;
	    case GLUT_KEY_DOWN:
		cross_sec_up_down -= step_sizey;
		if( cross_sec_up_down < AxisMin_y )
			cross_sec_up_down = AxisMin_y;
		break;
	    case GLUT_KEY_LEFT:
		cross_sec_left_right -= step_sizex;
		if( cross_sec_left_right < AxisMin_x )
			cross_sec_left_right = AxisMin_x;
		break;
	    case GLUT_KEY_RIGHT:
		cross_sec_left_right += step_sizex;
		if( cross_sec_left_right > AxisMax_x )
			cross_sec_left_right = AxisMax_x;
		break;
	}
  }
  else
  {

/* The arrow keys move the mesh vertically and horizontally
   in the window. */

	switch (key) {
	    case GLUT_KEY_UP:
		up_down += step_sizey;
		break;
	    case GLUT_KEY_DOWN:
		up_down -= step_sizey;
		break;
	    case GLUT_KEY_LEFT:
		left_right -= step_sizex;
		break;
	    case GLUT_KEY_RIGHT:
		left_right += step_sizex;
		break;
	}
  }

/* For the curvature */

  switch (key) {
	case GLUT_KEY_F1:
	    color_choice = 41;
	    break;
	case GLUT_KEY_F2:
	    color_choice = 42;
	    break;
	case GLUT_KEY_F3:
	    color_choice = 43;
	    break;
/* For the moments */
	case GLUT_KEY_F7:
	    color_choice = 50;
	    break;
	case GLUT_KEY_F8:
	    color_choice = 51;
	    break;
	case GLUT_KEY_F9:
	    color_choice = 52;
	    break;
  }
  glutPostRedisplay();
}


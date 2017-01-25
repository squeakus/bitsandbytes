/*
    This source file contains all the mesh window initialization functions
    for every FEM GUI program.

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

extern int Render_flag;
extern GLfloat mat_specular[4];
extern GLfloat mat_shininess[1];
extern GLfloat light_position[4];

extern double step_sizex, step_sizey, step_sizez;
extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double AxisMax_x, AxisMax_y, AxisMax_z, AxisMin_x, AxisMin_y, AxisMin_z;
extern double left, right, top, bottom, near, far, fscale;
extern double ortho_left, ortho_right, ortho_top, ortho_bottom;
extern int mesh_width, mesh_height;
extern int Perspective_flag;
extern int CrossSection_flag;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out,
        cross_sec_left_right0, cross_sec_up_down0, cross_sec_in_out0;

void MeshKey_Special(int key, int x, int y)
{
/*
    This program contains the mesh special keys routine for every
    FEM GUI program.  Look in:

      /usr/include/GL/glut.h

   for a list of the special keys.
  
	                Last Update 5/27/01
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
  glutPostRedisplay();
}

/*
 *  This program contains the mesh reshape routine for every FEM GUI
 *  program.
 *
 *                      Last Update 4/23/01
 */

void MeshReshape(int w, int h)
{
	double ratio;

	mesh_width = w;
	mesh_height = h;

	ratio = (double) mesh_width / (double) mesh_height;
	glViewport (0, 0, mesh_width, mesh_height);    /*  define the viewport */
	glMatrixMode (GL_PROJECTION);       /*  prepare for and then  */
	glLoadIdentity ();  
	/*glOrtho (-2.0, 2.0, -2.0, 2.0,1, 40.0); 
	glOrtho (left,right,bottom,top, near,1000);*/


	if(Perspective_flag)
	{
		glFrustum (-ratio, ratio, -1.0, 1.0, 2.0, 1000.0); 
		/*up_down = up_down0;*/
	}
	else
	{
		glOrtho ( ortho_left, ortho_right, ortho_bottom,
			ortho_top, near, 1000);
		/*up_down = up_down0;*/
	}
	
	/*glFrustum (-2.0, 2.0, -2.0, 2.0,1, 40.0); */
	/*glFrustum (left,right,bottom,top, near,far);*/
	glMatrixMode (GL_MODELVIEW);       /*  back to modelview matrix    */
	glLoadIdentity();
}

/*
 *  This program contains the mesh init routine for every FEM GUI
 *  program.
 *
 *                      Last Update 1/25/02
 */

void MeshInit(void)
{

   glLightfv(GL_LIGHT0, GL_POSITION, light_position);
   glEnable(GL_LIGHT0);
   glDepthFunc(GL_LEQUAL);
   glEnable(GL_DEPTH_TEST);
   glEnable (GL_BLEND);
   glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

/* I had to make the modification below because Mesa-3.4.1 changed
   the background color from light to dark grey with:

   glClearColor(.51, .51, .51, 1.0);
*/
   glClearColor(.5, .5, .5, 1.0);

   if(Render_flag)
   {

/*The lines below do shaded rendering*/

	glLoadIdentity ();  
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glShadeModel(GL_SMOOTH);
   }
   else
   {

/* Disable the Rendering Properties */

	glDisable(GL_LIGHTING);
    }
    glutPostRedisplay();
	
}


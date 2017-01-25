/*
    This program contains the mesh display routine for the FEM GUI
    for shell elements.
  
                  Last Update 9/28/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "../shell/shconst.h"
#include "../shell/shstruct.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

void shmeshdraw_triangle(void);
void shmeshdraw(void);
void shrender_triangle(void);
void shrender(void);
void shdisp_vectors(BOUND , double *);
void shforce_vectors(BOUND , double *, XYZPhiF *);

extern int flag_quad_element;
extern double *coord;
extern BOUND bc;

extern double left_right, up_down, in_out, left_right0,
	up_down0, in_out0, xAngle, yAngle, zAngle;
extern GLuint AxesList, DispList, ForceList;   /* Display lists */
extern XYZPhiF *force_vec;
extern int Render_flag, AppliedDisp_flag, AppliedForce_flag,
    Axes_flag, Before_flag, After_flag; 
extern int CrossSection_flag;

void AxesNumbers2(void);

void AxesNumbers(void);

void AxesLabel(void);

void CrossSetionPlaneDraw(void);

void shMeshDisplay(void)
{
	glClear (GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

	glLoadIdentity ();  /*  clear the matrix    */

	glTranslatef (left_right, up_down, in_out);
	glRotatef (xAngle, 1, 0, 0);
	glRotatef (yAngle, 0, 1, 0);
	glRotatef (zAngle, 0, 0, 1);

	glPointSize(8);
	if(Axes_flag)
		glCallList(AxesList);
	if(AppliedDisp_flag)
	{
		if(Before_flag )
			glCallList(DispList);
		if(After_flag )
			shdisp_vectors(bc, coord);
	}
	if(AppliedForce_flag)
	{
		if(Before_flag )
			glCallList(ForceList);
		if(After_flag )
			shforce_vectors(bc, coord, force_vec);
	}
	if(CrossSection_flag)
	{
		CrossSetionPlaneDraw();
	}
	glLineWidth (2.0);
	if(Render_flag)
	{
		if(flag_quad_element) shrender();
		else shrender_triangle();
	}
	else
	{
		if(flag_quad_element) shmeshdraw();
		else shmeshdraw_triangle();
	}
	if(Axes_flag)
	{
		AxesNumbers();
		/*AxesNumbers2();*/
	}
	glutSwapBuffers();
}


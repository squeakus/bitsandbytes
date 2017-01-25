/*
    This program contains the mesh display routine for the FEM GUI
    for beam elements.
  
                  Last Update 1/21/06

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

void bmmeshdraw(void);
void bmdist_load_vectors(BOUND , int *, double *, XYZF_GR * );
void bmdisp_vectors(BOUND , double *);
void bmforce_vectors(BOUND , double *, XYZPhiF *);

extern double *coord;
extern int *connecter;
extern BOUND bc;

extern double left_right, up_down, in_out, left_right0,
	up_down0, in_out0, xAngle, yAngle, zAngle;
extern GLuint AxesList, DispList, ForceList, Dist_LoadList;   /* Display lists */
extern XYZPhiF *force_vec;
extern XYZF_GR *dist_load_vec;
extern int Render_flag, AppliedDisp_flag, AppliedForce_flag, Dist_Load_flag,
    Axes_flag, Before_flag, After_flag; 
extern int CrossSection_flag;

void AxesNumbers2(void);

void AxesNumbers(void);

void AxesLabel(void);

void CrossSetionPlaneDraw(void);

void bmMeshDisplay(void)
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
			bmdisp_vectors(bc, coord);
	}
	if(AppliedForce_flag)
	{
		if(Before_flag )
			glCallList(ForceList);
		if(After_flag )
			bmforce_vectors(bc, coord, force_vec);
	}
	if(Dist_Load_flag)
	{
		if(Before_flag )
			glCallList(Dist_LoadList);
		if(After_flag )
			bmdist_load_vectors(bc, connecter, coord, dist_load_vec);
	}
	if(CrossSection_flag)
	{
		CrossSetionPlaneDraw();
	}
	glLineWidth (2.0);
	bmmeshdraw();
	if(Axes_flag)
	{
		AxesNumbers();
		/*AxesNumbers2();*/
	}
	glutSwapBuffers();
}


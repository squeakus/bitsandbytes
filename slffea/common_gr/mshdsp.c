/*
    This program contains the mesh display routine for the FEM GUI
    for 3-D elements.
  
                  Last Update 8/16/06

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

#if BRICK1
#include "../brick/brick/brconst.h"
#include "../brick/brick/brstruct.h"
#endif
#if BRICK2
#include "../brick/brick/brconst.h"
#include "../brick/brick2/br2struct.h"
#endif
#if QUAD1
#include "../quad/quad/qdconst.h"
#include "../quad/quad/qdstruct.h"
#endif
#if QUAD2
#include "../quad/quad/qdconst.h"
#include "../quad/quad2/qd2struct.h"
#endif
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#endif
#if TETRA2
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra2/te2struct.h"
#endif
#if TRI1
#include "../tri/tri/trconst.h"
#include "../tri/tri/trstruct.h"
#endif
#if TRI2
#include "../tri/tri/trconst.h"
#include "../tri/tri2/tr2struct.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#endif
#if WEDGE2
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge2/we2struct.h"
#endif


#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#if BRICK1 || BRICK2
void brmeshdraw(void);
void brrender(void);
#endif

#if QUAD1 || QUAD2
void qdmeshdraw(void);
void qdrender(void);
#endif

#if TETRA1 || TETRA2
void temeshdraw(void);
void terender(void);
#endif

#if TRI1 || TRI2
void trmeshdraw(void);
void trrender(void);
#endif

#if WEDGE1 || WEDGE2
void wemeshdraw(void);
void werender(void);
#endif

void disp_vectors(BOUND , double *);
void force_vectors(BOUND , double *, XYZF *);

extern double *coord;
extern BOUND bc;

extern double left_right, up_down, in_out, left_right0,
	up_down0, in_out0, xAngle, yAngle, zAngle;
extern GLuint AxesList, DispList, ForceList;   /* Display lists */
extern XYZF *force_vec;
extern int Render_flag, AppliedDisp_flag, AppliedForce_flag,
    Axes_flag, Before_flag, After_flag; 
extern int CrossSection_flag;

void AxesNumbers2(void);

void AxesNumbers(void);

void AxesLabel(void);

void CrossSetionPlaneDraw(void);

void MeshDisplay(void)
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
			disp_vectors(bc, coord);
	}
	if(AppliedForce_flag)
	{
		if(Before_flag )
			glCallList(ForceList);
		if(After_flag )
			force_vectors(bc, coord, force_vec);
	}
	if(CrossSection_flag)
	{
		CrossSetionPlaneDraw();
	}
	glLineWidth (2.0);

#if BRICK1 || BRICK2
	if(Render_flag)
		brrender();
	else
		brmeshdraw();
#endif
#if QUAD1 || QUAD2
	if(Render_flag)
		qdrender();
	else
		qdmeshdraw();
#endif
#if TETRA1 || TETRA2
	if(Render_flag)
		terender();
	else
		temeshdraw();
#endif
#if TRI1 || TRI2
	if(Render_flag)
		trrender();
	else
		trmeshdraw();
#endif
#if WEDGE1 || WEDGE2
	if(Render_flag)
		werender();
	else
		wemeshdraw();
#endif

	if(Axes_flag)
	{
		AxesNumbers();
		/*AxesNumbers2();*/
	}
	glutSwapBuffers();
}


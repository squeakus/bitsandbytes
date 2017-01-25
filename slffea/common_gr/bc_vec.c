/*
   This program draws the displacement and force vectors
   for 3-D elements that have no rotational degrees of freedom. 
 
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
#include <math.h>
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
#if TETRA1
#include "../tetra/tetra/teconst.h"
#include "../tetra/tetra/testruct.h"
#endif
#if TRI1
#include "../tri/tri/trconst.h"
#include "../tri/tri/trstruct.h"
#endif
#if TRUSS1
#include "../truss/truss/tsconst.h"
#include "../truss/truss/tsstruct.h"
#endif
#if WEDGE1
#include "../wedge/wedge/weconst.h"
#include "../wedge/wedge/westruct.h"
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern GLfloat yellow[4], orange[4], orangeRed[4], red[4], green[4], 
	violetRed[4], magenta[4], purple[4], blue[4],
	white[4], grey[4], black[4], brown[4];
extern double AxisLength_max;

void disp_vectors0(int displaylistnum, BOUND bc, double *coord0)
{

/* draws displacement vector on undefromed configuration */

  int i,j, dum;
  double fdum, fpointx, fpointy, fpointz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  fdum = AxisLength_max;
  glNewList(displaylistnum, GL_COMPILE);
  glPushAttrib(GL_LIGHTING_BIT);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (4.0);
    glBegin(GL_LINES);
/* Draw the x displacement vectors */
	glColor4fv(white);
	for( i = 0; i < bc.num_fix[0].x; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].x);
		fpointy = *(coord0+nsd*bc.fix[i].x + 1);
		fpointz = *(coord0+nsd*bc.fix[i].x + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum, fpointy, fpointz);
	}
/* Draw the y displacement vectors */
	glColor4fv(grey);
	for( i = 0; i < bc.num_fix[0].y; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].y);
		fpointy = *(coord0+nsd*bc.fix[i].y + 1);
		fpointz = *(coord0+nsd*bc.fix[i].y + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx, fpointy - fdum, fpointz);
	}
/* Draw the z displacement vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_fix[0].z; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].z);
		fpointy = *(coord0+nsd*bc.fix[i].z + 1);
		fpointz = *(coord0+nsd*bc.fix[i].z + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx, fpointy, fpointz - fdum);
	}
    glEnd();
    glPointSize(8);
    glBegin(GL_POINTS);
	glColor4fv(blue);
	for( i = 0; i < bc.num_fix[0].x; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].x);
		fpointy = *(coord0+nsd*bc.fix[i].x + 1);
		fpointz = *(coord0+nsd*bc.fix[i].x + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].y; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].y);
		fpointy = *(coord0+nsd*bc.fix[i].y + 1);
		fpointz = *(coord0+nsd*bc.fix[i].y + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].z; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].z);
		fpointy = *(coord0+nsd*bc.fix[i].z + 1);
		fpointz = *(coord0+nsd*bc.fix[i].z + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
  glPopAttrib();
  glEndList();
}

void force_vectors0(int displaylistnum, BOUND bc, double *coord0,
	XYZF *force_vec )
{

/* draws force vector on undefromed configuration */

  int i,j, dum;
  double fpointx, fpointy, fpointz, fx, fy, fz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  glNewList(displaylistnum, GL_COMPILE);
  glPushAttrib(GL_LIGHTING_BIT);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (4.0);
    glBegin(GL_LINES);
/* Draw the force vectors axis */
	glColor4fv(white);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
		fx = force_vec[i].x; fy = force_vec[i].y;
			fz = force_vec[i].z;
		fpointx = *(coord0+nsd*bc.force[i]);
		fpointy = *(coord0+nsd*bc.force[i] + 1);
		fpointz = *(coord0+nsd*bc.force[i] + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fx, fy, fz);
	}
    glEnd();
    glPointSize(8);
    glBegin(GL_POINTS);
	glColor4fv(red);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
		fpointx = *(coord0+nsd*bc.force[i]);
		fpointy = *(coord0+nsd*bc.force[i] + 1);
		fpointz = *(coord0+nsd*bc.force[i] + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
  glPopAttrib();
  glEndList();
}

void disp_vectors(BOUND bc, double *coord)
{

/* draws displacement vector on deformed configuration */

  int i,j, dum;
  double fdum, fpointx, fpointy, fpointz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  fdum = AxisLength_max;
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (4.0);
    glBegin(GL_LINES);
/* Draw the x displacement vectors */
	glColor4fv(white);
	for( i = 0; i < bc.num_fix[0].x; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].x);
		fpointy = *(coord+nsd*bc.fix[i].x + 1);
		fpointz = *(coord+nsd*bc.fix[i].x + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum, fpointy, fpointz);
	}
/* Draw the y displacement vectors */
	glColor4fv(grey);
	for( i = 0; i < bc.num_fix[0].y; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].y);
		fpointy = *(coord+nsd*bc.fix[i].y + 1);
		fpointz = *(coord+nsd*bc.fix[i].y + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx, fpointy - fdum, fpointz);
	}
/* Draw the z displacement vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_fix[0].z; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].z);
		fpointy = *(coord+nsd*bc.fix[i].z + 1);
		fpointz = *(coord+nsd*bc.fix[i].z + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx, fpointy, fpointz - fdum);
	}
    glEnd();
    glPointSize(8);
    glBegin(GL_POINTS);
	glColor4fv(blue);
	for( i = 0; i < bc.num_fix[0].x; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].x);
		fpointy = *(coord+nsd*bc.fix[i].x + 1);
		fpointz = *(coord+nsd*bc.fix[i].x + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].y; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].y);
		fpointy = *(coord+nsd*bc.fix[i].y + 1);
		fpointz = *(coord+nsd*bc.fix[i].y + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].z; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].z);
		fpointy = *(coord+nsd*bc.fix[i].z + 1);
		fpointz = *(coord+nsd*bc.fix[i].z + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
}

void force_vectors(BOUND bc, double *coord, XYZF *force_vec )
{

/* draws force vector on deformed configuration */

  int i,j, dum;
  double fpointx, fpointy, fpointz, fx, fy, fz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (4.0);
    glBegin(GL_LINES);
/* Draw the force vectors axis */
	glColor4fv(white);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
		fx = force_vec[i].x; fy = force_vec[i].y;
			fz = force_vec[i].z;
		fpointx = *(coord+nsd*bc.force[i]);
		fpointy = *(coord+nsd*bc.force[i] + 1);
		fpointz = *(coord+nsd*bc.force[i] + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fx, fy, fz);
	}
    glEnd();
    glPointSize(8);
    glBegin(GL_POINTS);
	glColor4fv(red);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
		fpointx = *(coord+nsd*bc.force[i]);
		fpointy = *(coord+nsd*bc.force[i] + 1);
		fpointz = *(coord+nsd*bc.force[i] + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
}


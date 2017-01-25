/*
    This program draws the displacement and force vectors
    for beam elements. 
 
                  Last Update 9/17/06

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"
#include "../../common_gr/control.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern GLfloat yellow[4], orange[4], orangeRed[4], red[4], green[4], 
	violetRed[4], magenta[4], purple[4], blue[4],
	white[4], grey[4], black[4], brown[4];
extern double AxisLength_max;

void bmdisp_vectors0(int displaylistnum, BOUND bc, double *coord0)
{

/* draws displacement vector on undefromed configuration */

  int i,j, dum;
  double fdum, fdum2, fpointx, fpointy, fpointz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  fdum = AxisLength_max;
  fdum2 = .5*AxisLength_max;
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
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the x angle vectors */
	glColor4fv(white);
	for( i = 0; i < bc.num_fix[0].phix; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phix);
		fpointy = *(coord0+nsd*bc.fix[i].phix + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phix + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx - fdum2, fpointy + fdum2, fpointz); 
		glVertex3f( fpointx - fdum2, fpointy + fdum2, fpointz); 
		glVertex3f( fpointx, fpointy, fpointz); 
	}
/* Draw the y angle vectors */
	glColor4fv(grey);
	for( i = 0; i < bc.num_fix[0].phiy; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phiy);
		fpointy = *(coord0+nsd*bc.fix[i].phiy + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phiy + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx + fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx + fdum2, fpointy - fdum2, fpointz); 
		glVertex3f( fpointx, fpointy, fpointz); 
	}
/* Draw the z angle vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_fix[0].phiz; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phiz);
		fpointy = *(coord0+nsd*bc.fix[i].phiz + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phiz + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx - fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx + fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx + fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx, fpointy, fpointz);
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
	for( i = 0; i < bc.num_fix[0].phix; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phix);
		fpointy = *(coord0+nsd*bc.fix[i].phix + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phix + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].phiy; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phiy);
		fpointy = *(coord0+nsd*bc.fix[i].phiy + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phiy + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].phiz; ++i)
	{
		fpointx = *(coord0+nsd*bc.fix[i].phiz);
		fpointy = *(coord0+nsd*bc.fix[i].phiz + 1);
		fpointz = *(coord0+nsd*bc.fix[i].phiz + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
  glPopAttrib();
  glEndList();
}

void bmforce_vectors0(int displaylistnum, BOUND bc, double *coord0,
	XYZPhiF *force_vec )
{

/* draws force vector on undefromed configuration */

  int i,j, dum;
  double fpointx, fpointy, fpointz, fx, fy, fz, fdum, fdum2, fdum3, fdum4;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  glNewList(displaylistnum, GL_COMPILE);
  glPushAttrib(GL_LIGHTING_BIT);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (1.0);
    glBegin(GL_LINES);
/* Draw the force vectors */
	glColor4fv(grey);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
	    fx = force_vec[i].x; fy = force_vec[i].y;
		fz = force_vec[i].z;
	    fpointx = *(coord0+nsd*bc.force[i]);
	    fpointy = *(coord0+nsd*bc.force[i] + 1);
	    fpointz = *(coord0+nsd*bc.force[i] + 2);
	    fdum = fabs(fx-fpointx);
	    fdum += fabs(fy-fpointy);
	    fdum += fabs(fz-fpointz);
	    if( fdum > SMALL)
	    {
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fx, fy, fz); 
	    }
	}
    glEnd();
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the moment vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
	    fx = force_vec[i].phix; fy = force_vec[i].phiy;
		fz = force_vec[i].phiz;
	    fpointx = *(coord0+nsd*bc.force[i]);
	    fpointy = *(coord0+nsd*bc.force[i] + 1);
	    fpointz = *(coord0+nsd*bc.force[i] + 2);
	    fdum = fx-fpointx;
	    fdum2 = fy-fpointy;
	    fdum3 = fz-fpointz;
	    fdum4 = fabs(fdum) + fabs(fdum2) + fabs(fdum3);
	    if( fdum4 > SMALL)
	    {
		fdum *= .2;
		fdum2 *= .2;
		fdum3 *= .2;
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fx + fdum2 + fdum3, fy - fdum - fdum3, fz + fdum - fdum2);
		glVertex3f( fx + fdum2 + fdum3, fy - fdum - fdum3, fz + fdum - fdum2);
		glVertex3f( fx - fdum2 - fdum3, fy + fdum + fdum3, fz - fdum + fdum2);
		glVertex3f( fx - fdum2 - fdum3, fy + fdum + fdum3, fz - fdum + fdum2);
		glVertex3f( fpointx, fpointy, fpointz);
	    }
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

void bmdist_load_vectors0(int displaylistnum, BOUND bc, int *connecter,
	double *coord0, XYZF_GR *dist_load_vec )
{

/* draws distributed load vectors on undeformed configuration */

  int i,j,k, dum;
  int node0, node1;
  double fpointx0, fpointy0, fpointz0, fpointx1, fpointy1, fpointz1,
	fx, fy, fz, el_dx, el_dy, el_dz, el_length,
	fdum1, fdum2, fdum3, fdum4, fdum5;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  glNewList(displaylistnum, GL_COMPILE);
  glPushAttrib(GL_LIGHTING_BIT);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the distributed load vectors */
	glColor4fv(grey);
	for( k = 0; k < bc.num_dist_load[0]; ++k)
	{
	    fx = dist_load_vec[k].x; fy = dist_load_vec[k].y;
		fz = dist_load_vec[k].z;

	    node0 = *(connecter+bc.dist_load[k]*npel);
	    node1 = *(connecter+bc.dist_load[k]*npel+1);

	    fpointx0 = *(coord0+nsd*node0);
	    fpointy0 = *(coord0+nsd*node0+1);
	    fpointz0 = *(coord0+nsd*node0+2);

	    fpointx1 = *(coord0+nsd*node1);
	    fpointy1 = *(coord0+nsd*node1+1);
	    fpointz1 = *(coord0+nsd*node1+2);

	    glVertex3f( fpointx0, fpointy0, fpointz0);
	    glVertex3f( fpointx0 - fx, fpointy0 + fy, fpointz0 + fz); 
	    glVertex3f( fpointx0 - fx, fpointy0 + fy, fpointz0 + fz); 
	    glVertex3f( fpointx1 - fx, fpointy1 + fy, fpointz1 + fz); 
	    glVertex3f( fpointx1 - fx, fpointy1 + fy, fpointz1 + fz); 
	    glVertex3f( fpointx1, fpointy1, fpointz1);
	    fdum1 = .66*fpointx0+.33*fpointx1;
	    fdum2 = .66*fpointy0+.33*fpointy1;
	    fdum3 = .66*fpointz0+.33*fpointz1;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .66*fpointx0+.33*fpointx1 - fx;
	    fdum2 = .66*fpointy0+.33*fpointy1 + fy;
	    fdum3 = .66*fpointz0+.33*fpointz1 + fz;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .33*fpointx0+.66*fpointx1;
	    fdum2 = .33*fpointy0+.66*fpointy1;
	    fdum3 = .33*fpointz0+.66*fpointz1;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .33*fpointx0+.66*fpointx1 - fx;
	    fdum2 = .33*fpointy0+.66*fpointy1 + fy;
	    fdum3 = .33*fpointz0+.66*fpointz1 + fz;
	    glVertex3f( fdum1, fdum2, fdum3); 
	}
    glEnd();
    glPointSize(8);
    glBegin(GL_POINTS);
	glColor4fv(white);
	for( k = 0; k < bc.num_dist_load[0]; ++k)
	{
	    node0 = *(connecter+bc.dist_load[k]*npel);
	    node1 = *(connecter+bc.dist_load[k]*npel+1);

	    fpointx0 = *(coord0+nsd*node0);
	    fpointy0 = *(coord0+nsd*node0+1);
	    fpointz0 = *(coord0+nsd*node0+2);

	    fpointx1 = *(coord0+nsd*node1);
	    fpointy1 = *(coord0+nsd*node1+1);
	    fpointz1 = *(coord0+nsd*node1+2);

	    glVertex3f( fpointx0, fpointy0, fpointz0);
	    glVertex3f( fpointx1, fpointy1, fpointz1);
	}
    glEnd();
    glPointSize(1);

  glPopAttrib();
  glEndList();
}

void bmdisp_vectors(BOUND bc, double *coord)
{

/* AFFECTS AXES.draws displacement vector on deformed configuration */

  int i,j, dum;
  double fdum, fdum2, fpointx, fpointy, fpointz;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  fdum = AxisLength_max;
  fdum2 = .5*AxisLength_max;
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
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the x angle vectors */
	glColor4fv(white);
	for( i = 0; i < bc.num_fix[0].phix; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phix);
		fpointy = *(coord+nsd*bc.fix[i].phix + 1);
		fpointz = *(coord+nsd*bc.fix[i].phix + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx - fdum2, fpointy + fdum2, fpointz);
		glVertex3f( fpointx - fdum2, fpointy + fdum2, fpointz);
		glVertex3f( fpointx, fpointy, fpointz);
	}
/* Draw the y angle vectors */
	glColor4fv(grey);
	for( i = 0; i < bc.num_fix[0].phiy; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phiy);
		fpointy = *(coord+nsd*bc.fix[i].phiy + 1);
		fpointz = *(coord+nsd*bc.fix[i].phiy + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx - fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx + fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx + fdum2, fpointy - fdum2, fpointz);
		glVertex3f( fpointx, fpointy, fpointz);
	}
/* Draw the z angle vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_fix[0].phiz; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phiz);
		fpointy = *(coord+nsd*bc.fix[i].phiz + 1);
		fpointz = *(coord+nsd*bc.fix[i].phiz + 2);
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fpointx - fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx - fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx + fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx + fdum2, fpointy, fpointz - fdum2);
		glVertex3f( fpointx, fpointy, fpointz);
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
	for( i = 0; i < bc.num_fix[0].phix; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phix);
		fpointy = *(coord+nsd*bc.fix[i].phix + 1);
		fpointz = *(coord+nsd*bc.fix[i].phix + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].phiy; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phiy);
		fpointy = *(coord+nsd*bc.fix[i].phiy + 1);
		fpointz = *(coord+nsd*bc.fix[i].phiy + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
	for( i = 0; i < bc.num_fix[0].phiz; ++i)
	{
		fpointx = *(coord+nsd*bc.fix[i].phiz);
		fpointy = *(coord+nsd*bc.fix[i].phiz + 1);
		fpointz = *(coord+nsd*bc.fix[i].phiz + 2);
		glVertex3f( fpointx, fpointy, fpointz);
	}
    glEnd();
    glPointSize(1);
}

void bmforce_vectors(BOUND bc, double *coord, XYZPhiF *force_vec )
{

/* THIS DRAWS THE FORCE,draws force vector on deformed configuration */

  int i,j, dum;
  double fpointx, fpointy, fpointz, fx, fy, fz, fdum, fdum2, fdum3, fdum4;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (1.0);
    glBegin(GL_LINES);
/* Draw the force vectors */
	glColor4fv(white);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
	    fx = force_vec[i].x; fy = force_vec[i].y;
		fz = force_vec[i].z;
	    fpointx = *(coord+nsd*bc.force[i]);
	    fpointy = *(coord+nsd*bc.force[i] + 1);
	    fpointz = *(coord+nsd*bc.force[i] + 2);
	    fdum = fabs(fx-fpointx);
	    fdum += fabs(fy-fpointy);
	    fdum += fabs(fz-fpointz);
	    fz = fz - ((fz-fpointz) * 0.7);
	    if( fdum > SMALL)
	    {
	      /*printf("ax: %f ay: %f az %f\n",fpointx,fpointy,fpointz);
		printf("bx: %f by: %f bz %f\n\n",fx,fy,fz);*/
	      glVertex3f( fpointx, fpointy, fpointz);
	      glVertex3f( fx, fy, fz); 
		
	    }
	}
    glEnd();
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the moment vectors */
	glColor4fv(black);
	for( i = 0; i < bc.num_force[0]; ++i)
	{
	    fx = force_vec[i].phix; fy = force_vec[i].phiy;
		fz = force_vec[i].phiz;
	    fpointx = *(coord+nsd*bc.force[i]);
	    fpointy = *(coord+nsd*bc.force[i] + 1);
	    fpointz = *(coord+nsd*bc.force[i] + 2);
	    fdum = fx-fpointx;
	    fdum2 = fy-fpointy;
	    fdum3 = fz-fpointz;
	    fdum4 = fabs(fdum) + fabs(fdum2) + fabs(fdum3);
	    if( fdum4 > SMALL)
	    {
		fdum *= .2;
		fdum2 *= .2;
		fdum3 *= .2;
		glVertex3f( fpointx, fpointy, fpointz);
		glVertex3f( fx + fdum2 + fdum3, fy - fdum - fdum3, fz + fdum - fdum2);
		glVertex3f( fx + fdum2 + fdum3, fy - fdum - fdum3, fz + fdum - fdum2);
		glVertex3f( fx - fdum2 - fdum3, fy + fdum + fdum3, fz - fdum + fdum2);
		glVertex3f( fx - fdum2 - fdum3, fy + fdum + fdum3, fz - fdum + fdum2);
		glVertex3f( fpointx, fpointy, fpointz);
	    }
	}
    glEnd();
    glPointSize(6);
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
    glPointSize(1);

}

void bmdist_load_vectors(BOUND bc, int *connecter, double *coord, XYZF_GR *dist_load_vec )
{

/* draws distributed load vectors on deformed configuration */

  int i,j,k, dum;
  int node0, node1;
  double fpointx0, fpointy0, fpointz0, fpointx1, fpointy1, fpointz1,
	fx, fy, fz, el_dx, el_dy, el_dz, el_length,
	fdum1, fdum2, fdum3, fdum4, fdum5;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glLineWidth (2.0);
    glBegin(GL_LINES);
/* Draw the distributed load vectors */
	glColor4fv(grey);
	for( k = 0; k < bc.num_dist_load[0]; ++k)
	{
	    fx = dist_load_vec[k].x; fy = dist_load_vec[k].y;
		fz = dist_load_vec[k].z;

	    node0 = *(connecter+bc.dist_load[k]*npel);
	    node1 = *(connecter+bc.dist_load[k]*npel+1);

	    fpointx0 = *(coord+nsd*node0);
	    fpointy0 = *(coord+nsd*node0+1);
	    fpointz0 = *(coord+nsd*node0+2);

	    fpointx1 = *(coord+nsd*node1);
	    fpointy1 = *(coord+nsd*node1+1);
	    fpointz1 = *(coord+nsd*node1+2);

	    glVertex3f( fpointx0, fpointy0, fpointz0);
	    glVertex3f( fpointx0 - fx, fpointy0 + fy, fpointz0 + fz); 
	    glVertex3f( fpointx0 - fx, fpointy0 + fy, fpointz0 + fz); 
	    glVertex3f( fpointx1 - fx, fpointy1 + fy, fpointz1 + fz); 
	    glVertex3f( fpointx1 - fx, fpointy1 + fy, fpointz1 + fz); 
	    glVertex3f( fpointx1, fpointy1, fpointz1);
	    fdum1 = .66*fpointx0+.33*fpointx1;
	    fdum2 = .66*fpointy0+.33*fpointy1;
	    fdum3 = .66*fpointz0+.33*fpointz1;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .66*fpointx0+.33*fpointx1 - fx;
	    fdum2 = .66*fpointy0+.33*fpointy1 + fy;
	    fdum3 = .66*fpointz0+.33*fpointz1 + fz;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .33*fpointx0+.66*fpointx1;
	    fdum2 = .33*fpointy0+.66*fpointy1;
	    fdum3 = .33*fpointz0+.66*fpointz1;
	    glVertex3f( fdum1, fdum2, fdum3); 
	    fdum1 = .33*fpointx0+.66*fpointx1 - fx;
	    fdum2 = .33*fpointy0+.66*fpointy1 + fy;
	    fdum3 = .33*fpointz0+.66*fpointz1 + fz;
	    glVertex3f( fdum1, fdum2, fdum3); 
	}
    glEnd();
    glPointSize(6);
    glBegin(GL_POINTS);
	glColor4fv(white);
	for( k = 0; k < bc.num_dist_load[0]; ++k)
	{
	    node0 = *(connecter+bc.dist_load[k]*npel);
	    node1 = *(connecter+bc.dist_load[k]*npel+1);

	    fpointx0 = *(coord+nsd*node0);
	    fpointy0 = *(coord+nsd*node0+1);
	    fpointz0 = *(coord+nsd*node0+2);

	    fpointx1 = *(coord+nsd*node1);
	    fpointy1 = *(coord+nsd*node1+1);
	    fpointz1 = *(coord+nsd*node1+2);

	    glVertex3f( fpointx0, fpointy0, fpointz0);
	    glVertex3f( fpointx1, fpointy1, fpointz1);
	}
    glEnd();
    glPointSize(1);
}


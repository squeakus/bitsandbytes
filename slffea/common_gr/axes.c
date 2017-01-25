/*
    This program draws the axes using a display list. 
    It is modeled on the axes routine in 
    Philip Winston's program "walker"( specifically,
    the agvMakeAxesList subroutine).  He has graciously
    and generously allowed me to use and modify it for these finite
    element graphics programs.
  
                                  San Le
  
                  Last Update 1/25/06
 
    You can reach him at:
  
    Philip Winston - 4/11/95
    winston@cs.unc.edu
    http://www.cs.hmc.edu/people/pwinston
  
    SLFFEA source file
    Version:  1.5

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern GLfloat yellow[4], orange[4], orangeRed[4], red[4], green[4], 
	violetRed[4], magenta[4], purple[4], blue[4],
	white[4], grey[4], black[4];

extern double AxisMax_x, AxisMax_y, AxisMax_z,
	AxisMin_x, AxisMin_y, AxisMin_z, IAxisMin_x, IAxisMin_y, IAxisMin_z;
extern double AxisLength_x, AxisLength_y, AxisLength_z, AxisLength_max, AxisPoint_step;

void agvMakeAxesList(GLuint displaylistnum)
{
  int i,j, dum;
  double fdum, fdum2, fdum3, textmove;
  GLfloat axes_ambuse[] =   { 0.5, 0.0, 0.0, 1.0 };
  glNewList(displaylistnum, GL_COMPILE);
  glLineWidth (2.0);
  glPushAttrib(GL_LIGHTING_BIT);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, axes_ambuse);
    glBegin(GL_LINES);
	glColor4fv(white);
/* Draw the x axis */
      glVertex3f(AxisMax_x, 0, 0); glVertex3f(AxisMin_x, 0, 0);
	glColor4fv(grey);
/* Draw the y axis */
      glVertex3f(0, AxisMax_y, 0); glVertex3f(0, AxisMin_y, 0);
	glColor4fv(black);
/* Draw the z axis */
      glVertex3f(0, 0, AxisMax_z); glVertex3f(0, 0, AxisMin_z);
    glEnd();
    glPointSize(1);
    glBegin(GL_POINTS);
/* Draw the x axis points */
	dum = (int) (AxisLength_x/AxisPoint_step);
	glBegin(GL_POINTS);
	glColor4fv(black);
	fdum = IAxisMin_x;
	for( i = 0; i < dum; ++i)
	{
		glVertex3f( fdum, 0, 0);
		fdum += AxisPoint_step;
	}
/* Draw the y axis points */
	dum = (int) (AxisLength_y/AxisPoint_step);
	glBegin(GL_POINTS);
	glColor4fv(black);
	fdum = IAxisMin_y;
	for( i = 0; i < dum; ++i)
	{
		glVertex3f( 0, fdum, 0);
		fdum += AxisPoint_step;
	}
/* Draw the z axis points */
	dum = (int) (AxisLength_z/AxisPoint_step);
	glBegin(GL_POINTS);
	glColor4fv(black);
	fdum = IAxisMin_z;
	for( i = 0; i < dum; ++i)
	{
		glVertex3f( 0, 0, fdum);
		fdum += AxisPoint_step;
	}
    glEnd();
/* Draw the x axis label */
	textmove = AxisMax_x + 1.5*AxisLength_max;
	fdum = AxisLength_max + textmove;
	glBegin(GL_LINES);
	  glColor4fv(white);
	  glVertex3f(fdum, 0, 0); glVertex3f(textmove, 0, AxisLength_max);
	  glVertex3f(textmove, 0, 0); glVertex3f(fdum, 0, AxisLength_max);
	glEnd();
/* Draw the y axis label */
	textmove = AxisMax_y + 1.5*AxisLength_max;
	fdum = .5*AxisLength_max + textmove;
	fdum2 = .5*AxisLength_max;
	fdum3 = AxisLength_max + textmove;
	glBegin(GL_LINES);
	  glColor4fv(grey);
	  glVertex3f(0, textmove, 0); glVertex3f(0, fdum, 0);
	  glVertex3f(0, fdum, 0); glVertex3f(fdum2, fdum3, 0);
	  glVertex3f(0, fdum, 0); glVertex3f(-fdum2, fdum3, 0);
	glEnd();
/* Draw the z axis label */
	textmove = AxisMax_z + 1.5*AxisLength_max;
	fdum = AxisLength_max + textmove;
	glBegin(GL_LINES);
	  glColor4fv(black);
	  glVertex3f(0, 0, textmove); glVertex3f(AxisLength_max, 0, textmove);
	  glVertex3f(AxisLength_max, 0, textmove); glVertex3f(0, 0, fdum);
	  glVertex3f(0, 0, fdum); glVertex3f(AxisLength_max, 0, fdum);
	glEnd();
  glPopAttrib();
  glEndList();
}


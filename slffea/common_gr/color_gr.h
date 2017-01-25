/*
    This is the include file "color_gr.h" which contains the colors
    used in the graphics programs for the FEM code. 

                  Last Update 1/25/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "control.h"

/* glut header files */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#define SLFFEA                0    /* For Colors based on SLFFEA */
#define ANSYS                 1    /* For Colors based on ANSYS */
#define ANSYS_2               0    /* For Colors based on ANSYS_2*/

#if SLFFEA
GLfloat MeshColor[boxnumber+5][4] = {
	0.0, 0.0, 1.0, 1.0,
	0.627, 0.1255, 0.941, 1.0,
	1.0, 0.0, 1.0, 1.0,
	0.8157, 0.1255, 0.5647, 1.0,
	1.0, 0, 0, 1.0,
	1.0, 0.55, 0, 1.0,
	1.0, 0.647, 0, 1.0,
	1.0, 1.0, 0, 1.0,
	0.0, 1.0, 0.0, 1.0,
	1.0, 1.0, 1.0, 1.0,
	0.75, 0.75, 0.75, 1.0,
	0.5, 0.5, 0.5, 1.0,
	0.0, 0.0, 0.0, 1.0
};
#endif

#if ANSYS
GLfloat MeshColor[boxnumber+5][4] = {
	0.0,  0.0,  1.0, 1.0,
	0.4, 0.8,  1.0, 1.0,
	0.0,  1.0,  0.85, 1.0,
	0.0,  1.0,  0.0, 1.0,
	1.0,  1.0,  0, 1.0,
	1.0,  0.70,  0, 1.0,
	1.0,  0.54,  0, 1.0,
	1.0, 0, 0, 1.0,
	1.0, 0.0, 1.0, 1.0,
	1.0, 1.0, 1.0, 1.0,
	0.75, 0.75, 0.75, 1.0,
	0.5, 0.5, 0.5, 1.0,
	0.0, 0.0, 0.0, 1.0
};
#endif

#if ANSYS_2
GLfloat MeshColor[boxnumber+5][4] = {
	0.0,  0.0,  1.0, 1.0,
	0.3, 0.75,  0.9, 1.0,
	0.0,  1.0,  0.8, 1.0,
	0.0,  1.0,  0.0, 1.0,
	0.7,  1.0,  0, 1.0,
	1.0,  1.0,  0, 1.0,
	1.0,  0.7,  0, 1.0,
	1.0, 0, 0, 1.0,
	1.0, 0.0, 1.0, 1.0,
	1.0, 1.0, 1.0, 1.0,
	0.75, 0.75, 0.75, 1.0,
	0.5, 0.5, 0.5, 1.0,
	0.0, 0.0, 0.0 1.0,
};
#endif


GLfloat RenderColor[4] = { 0.5, 1.0, 1.0, 0.0};

GLfloat white[4] = {  1.0, 1.0, 1.0, 1.0 };
GLfloat grey[4] = { 0.75, 0.75, 0.75, 1.0 };
GLfloat darkGrey[4] = { 0.5, 0.5, 0.5, 1.0 };
GLfloat black[4] = { 0.0, 0.0, 0.0, 1.0 };
GLfloat green[4] = { 0.0, 1.0, 0.0, 1.0 };
GLfloat brown[4] = { 1.0, 0.2545, 0.2545, 1.0 };
GLfloat wire_color[4] = { 0.0, 0.0, 0.0, 1.0 };

GLfloat yellow[4] = { 1.0, 1.0, 0, 1.0 };
GLfloat orange[4] = { 1.0, 0.647, 0, 1.0 };
GLfloat orangeRed[4] = { 1.0, 0.55, 0, 1.0 };
GLfloat red[4] = { 1.0, 0, 0, 1.0 }; 
GLfloat violetRed[4] = { 0.8157, 0.1255, 0.5647, 1.0 };
GLfloat magenta[4] = { 1.0, 0.0, 1.0, 1.0 };
GLfloat purple[4] = { 0.627, 0.1255, 0.941, 1.0 };
GLfloat blue[4] = { 0.0, 0.0, 1.0, 1.0 };

GLfloat yellowRed[4] = { 1.0,  0.70,  0, 1.0 };
GLfloat greenYellow[4] = { 0.7, 1.0, 0.0, 1.0 };
GLfloat blueGreen[4] = { 0.0,  1.0,  0.85, 1.0 };
GLfloat aqua[4] = { 0.433, 0.837,  1.0, 1.0 };

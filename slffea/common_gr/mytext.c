
/* Copyright (c) Mark J. Kilgard, 1994. */

/*
 * (c) Copyright 1993, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 */
/*
 *  stroke.c
 *  This program demonstrates some characters of a
 *  stroke (vector) font.  The characters are represented
 *  by display lists, which are given numbers which
 *  correspond to the ASCII values of the characters.
 *  Use of glCallLists() is demonstrated.
 */

/* This code has been modified by San Le

	Update 5/4/01

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glut.h>
#include "mytext.h"


/*  drawLetter() interprets the instructions from the array
 *  for that letter and renders the letter with line segments.
 */
void drawLetter(CP *l)
{
    glBegin(GL_LINE_STRIP);
    for (;;) {
	switch (l->type) {
	    case PT:
		glVertex2fv(&l->x);
		break;
	    case STROKE:
		glVertex2fv(&l->x);
		glEnd();
		glBegin(GL_LINE_STRIP);
		break;
	    case END:
		glVertex2fv(&l->x);
		glEnd();
		glTranslatef(8.0, 0.0, 0.0);
		return;
	}
	l++;
    }
}

/*  Create a display list for each of 6 characters */
void make_character_lists (void)
{
    GLuint base;

    glShadeModel (GL_FLAT);

    base = glGenLists (128);
    glListBase(base);
    glNewList(base+'A', GL_COMPILE); drawLetter(Adata); glEndList();
    glNewList(base+'B', GL_COMPILE); drawLetter(Bdata); glEndList();
    glNewList(base+'C', GL_COMPILE); drawLetter(Cdata); glEndList();
    glNewList(base+'D', GL_COMPILE); drawLetter(Ddata); glEndList();
    glNewList(base+'E', GL_COMPILE); drawLetter(Edata); glEndList();
    glNewList(base+'e', GL_COMPILE); drawLetter(edata); glEndList();
    glNewList(base+'F', GL_COMPILE); drawLetter(Fdata); glEndList();
    glNewList(base+'G', GL_COMPILE); drawLetter(Gdata); glEndList();
    glNewList(base+'H', GL_COMPILE); drawLetter(Hdata); glEndList();
    glNewList(base+'I', GL_COMPILE); drawLetter(Idata); glEndList();
    glNewList(base+'J', GL_COMPILE); drawLetter(Jdata); glEndList();
    glNewList(base+'K', GL_COMPILE); drawLetter(Kdata); glEndList();
    glNewList(base+'L', GL_COMPILE); drawLetter(Ldata); glEndList();
    glNewList(base+'M', GL_COMPILE); drawLetter(Mdata); glEndList();
    glNewList(base+'N', GL_COMPILE); drawLetter(Ndata); glEndList();
    glNewList(base+'O', GL_COMPILE); drawLetter(Odata); glEndList();
    glNewList(base+'P', GL_COMPILE); drawLetter(Pdata); glEndList();
    glNewList(base+'Q', GL_COMPILE); drawLetter(Qdata); glEndList();
    glNewList(base+'R', GL_COMPILE); drawLetter(Rdata); glEndList();
    glNewList(base+'S', GL_COMPILE); drawLetter(Sdata); glEndList();
    glNewList(base+'T', GL_COMPILE); drawLetter(Tdata); glEndList();
    glNewList(base+'U', GL_COMPILE); drawLetter(Udata); glEndList();
    glNewList(base+'V', GL_COMPILE); drawLetter(Vdata); glEndList();
    glNewList(base+'W', GL_COMPILE); drawLetter(Wdata); glEndList();
    glNewList(base+'X', GL_COMPILE); drawLetter(Xdata); glEndList();
    glNewList(base+'Y', GL_COMPILE); drawLetter(Ydata); glEndList();
    glNewList(base+'Z', GL_COMPILE); drawLetter(Zdata); glEndList();
    glNewList(base+'1', GL_COMPILE); drawLetter(ndata1); glEndList();
    glNewList(base+'2', GL_COMPILE); drawLetter(ndata2); glEndList();
    glNewList(base+'3', GL_COMPILE); drawLetter(ndata3); glEndList();
    glNewList(base+'4', GL_COMPILE); drawLetter(ndata4); glEndList();
    glNewList(base+'5', GL_COMPILE); drawLetter(ndata5); glEndList();
    glNewList(base+'6', GL_COMPILE); drawLetter(ndata6); glEndList();
    glNewList(base+'7', GL_COMPILE); drawLetter(ndata7); glEndList();
    glNewList(base+'8', GL_COMPILE); drawLetter(ndata8); glEndList();
    glNewList(base+'9', GL_COMPILE); drawLetter(ndata9); glEndList();
    glNewList(base+'0', GL_COMPILE); drawLetter(ndata0); glEndList();
    glNewList(base+'+', GL_COMPILE); drawLetter(ndataplus); glEndList();
    glNewList(base+'-', GL_COMPILE); drawLetter(ndataminus); glEndList();
    glNewList(base+'.', GL_COMPILE); drawLetter(ndatadot); glEndList();
    glNewList(base+' ', GL_COMPILE); glTranslatef(8.0, 0.0, 0.0); glEndList();
}

#if 1
char *test1 = "A SPARE SERAPE APPEARS AS";
char *test2 = "APES PREPARE RARE PEPPERS";
char test3[40];
char test4[40];
#endif


void printStrokedString(char *s)
{
    GLsizei len = (GLsizei) strlen(s);
    glCallLists(len, GL_BYTE, (GLbyte *)s);
}

#if 1
void my_display(void)
{
    strncpy(test3,"ABCDEeFGHIJKLMNOPQRSTUVWXYZ1234567890+-.",40);
    sprintf( test4, "%10.3e ", 1422.39292);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);
    glPushMatrix();
    glScalef(2.0, 2.0, 2.0);
    glTranslatef(10.0, 30.0, 0.0);
    printStrokedString(test1);
    glPopMatrix();
    glPushMatrix();
    glScalef(2.0, 2.0, 2.0);
    glTranslatef(10.0, 13.0, 0.0);
    printStrokedString(test3);
    glPopMatrix();
    glPushMatrix();
    glScalef(2.0, 2.0, 2.0);
    glTranslatef(10.0, 60.0, 0.0);
    printStrokedString(test4);
    glFlush();
}
#endif

#if 0
static void reshape(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (GLdouble)w, 0.0, (GLdouble)h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void
key(unsigned char k, int x, int y)
{
  switch (k) {
  case 27:  /* Escape */
    exit(0);
    break;
  default:
    return;
  }
  glutPostRedisplay();
}

/*  Main Loop
 *  Open window with initial window size, title bar,
 *  RGBA display mode, and handle input events.
 */
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize (740, 220);
    glutCreateWindow (argv[0]);
    myinit ();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(key);
    glutMainLoop();
    return 0;             /* ANSI C requires main to return int. */
}
#endif

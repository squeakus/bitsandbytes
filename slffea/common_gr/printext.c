/*  This program contains the print text routine for every FEM GUI
    program.  It was based on code provided by Brendan J. Green
    from a posting he made to Usenet on the news group:
  
                comp.graphics.api.opengl
  
                     San Le
  
                        Last Update 5/14/00
  
    You can reach him at:
  
    Brendan J. Green 
    E-Mail:   bgreen@cs.rmit.edu.au
    Web Page: http://yallara.cs.rmit.edu.au/~bgreen
 
    SLFFEA source file
    Version:  1.5

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

void printText(const char* string)
{
	int index = 0;

	while (string[index] != '\0')
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, string[index]);
		index++;
	}
}


/*
    This program does a screen dump of the Mesh Window.  It
    is almost entirely based on a Usenet posting of code
    by Antony Searle on the group:
  
        comp.graphics.api.opengl
  
                                  San Le
  
                        Last Update 5/14/00
  
    You can reach him at:
  
    Antony Searle
    H1-NF National Plasma Fusion Research Facility
    Australian National University
    acs654@my-dejanews.com
  
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

void ScreenShot( int width, int height)
{
   unsigned char *ScreenBuffer;
    FILE *Handle;
    unsigned char Header[18];
	int dum;

/* The width and height have to be multiples of 20.  Targa seems to
   require this, or else rescaling of the box will result in
   distortion of the image
*/
	dum = width%4;
	width -=  dum;
	dum = height%4;
	height -= dum;

        /* use glReadPixels and save a .tga file */
        ScreenBuffer = (unsigned char *)
                calloc(3*4*width*height,sizeof(unsigned char));
        glReadPixels(0, 0, width, height, GL_BGR,
                GL_UNSIGNED_BYTE, ScreenBuffer);

    Header[ 0] = 0;
    Header[ 1] = 0;
    Header[ 2] = 2;     /* Uncompressed, uninteresting */
    Header[ 3] = 0;
    Header[ 4] = 0;
    Header[ 5] = 0;
    Header[ 6] = 0;
    Header[ 7] = 0;
    Header[ 8] = 0;
    Header[ 9] = 0;
    Header[10] = 0;
    Header[11] = 0;
    Header[12] = (unsigned char) width;  /* Dimensions */
    Header[13] = (unsigned char) ((unsigned long) width >> 8);
    Header[14] = (unsigned char) height;
    Header[15] = (unsigned char) ((unsigned long) height >> 8);
    Header[16] = 24;    /* Bits per pixel */
    Header[17] = 0;

    Handle = fopen("Screen.tga", "wb");
    if(Handle == NULL) {
                free(ScreenBuffer);
        return;
        }

    fseek(Handle, 0, 0);
    fwrite(Header, 1, 18, Handle);
    fseek(Handle, 18, 0);
    fwrite(ScreenBuffer, 3, width * height, Handle);
    fclose(Handle);

    free(ScreenBuffer);
}


/*
 * Demo of off-screen Mesa rendering
 *
 * See Mesa/include/GL/osmesa.h for documentation of the OSMesa functions.
 *
 * If you want to render BIG images you'll probably have to increase
 * MAX_WIDTH and MAX_HEIGHT in src/config.h.
 *
 * Brian Paul
 *
 * PPM output provided by Joerg Schmalzl.
 * ASCII PPM output added by Brian Paul.

  This file had been modified by San Le.

        Updated 4/2/01
  
    SLFFEA source file
    Version:  1.5

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 */


void ScreenShot_ppm( int width, int height)
{
   GLubyte *ScreenBuffer;
   const int binary = 1;
   FILE *f = fopen( "Screen.ppm", "w" );

   /* use glReadPixels and save a .tga file */
   ScreenBuffer = ( GLubyte *)
         calloc(4*width*height,sizeof( GLubyte ));
   glReadPixels(0, 0, width, height, GL_RGBA,
         GL_UNSIGNED_BYTE, ScreenBuffer);

   if (f) {
      int i, x, y;
      const GLubyte *ptr = ScreenBuffer;
      if (binary) {
         fprintf(f,"P6\n");
         fprintf(f,"# ppm-file created by osdemo.c\n");
         fprintf(f,"%i %i\n", width,height);
         fprintf(f,"255\n");
         fclose(f);
         f = fopen( "Screen.ppm", "ab" );  /* reopen in binary append mode */
         for (y=height-1; y>=0; y--) {
            for (x=0; x<width; x++) {
               i = (y*width + x) * 4;
               fputc(ptr[i], f);   /* write red */
               fputc(ptr[i+1], f); /* write green */
               fputc(ptr[i+2], f); /* write blue */
            }
         }
      }
      else {
         /*ASCII*/
         int counter = 0;
         fprintf(f,"P3\n");
         fprintf(f,"# ascii ppm file created by osdemo.c\n");
         fprintf(f,"%i %i\n", width, height);
         fprintf(f,"255\n");
         for (y=height-1; y>=0; y--) {
            for (x=0; x<width; x++) {
               i = (y*width + x) * 4;
               fprintf(f, " %3d %3d %3d", ptr[i], ptr[i+1], ptr[i+2]);
               counter++;
               if (counter % 5 == 0)
                  fprintf(f, "\n");
            }
         }
      }
      fclose(f);
   }
   free(ScreenBuffer);
}


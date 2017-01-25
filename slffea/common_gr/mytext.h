
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
#include <string.h>
#include <GL/glut.h>

#define PT 1
#define STROKE 2
#define END 3

typedef struct charpoint {
    GLfloat   x, y;
    int    type;
} CP;

CP Adata[] = {
    { 0, 0, PT}, {0, 9, PT}, {1, 10, PT}, {4, 10, PT},
    {5, 9, PT}, {5, 0, STROKE}, {0, 5, PT}, {5, 5, END}
};

CP Bdata[] = {
    {0, 0, PT}, {0, 10, PT},  {4, 10, PT}, {5, 9, PT}, {5, 6, PT},
    {4, 5, PT}, {0, 5, STROKE}, {4, 5, PT}, {5, 3, PT}, {5, 1, PT},
    {4, 0, PT}, {0, 0, END}
};

CP Cdata[] = {
    {5, 1, PT}, {4, 0, PT}, {1, 0, PT}, {0, 1, PT}, {0, 9, PT}, {1, 10, PT},
    {4, 10, PT}, {5, 9, END}
};

CP Ddata[] = {
    {0, 0, PT}, {0, 10, PT},  {4, 10, PT}, {5, 9, PT}, {5, 1, PT},
    {4, 0, PT}, {0, 0, END}
};

CP Edata[] = {
    {5, 0, PT}, {0, 0, PT}, {0, 10, PT}, {5, 10, STROKE},
    {0, 5, PT}, {4, 5, END}
};

CP edata[] = {
    {5, 1, PT}, {4, 0, PT}, {1, 0, PT}, {0, 1, PT}, {0, 6, PT}, {1, 7, PT},
    {4, 7, PT}, {5, 6, PT}, {5, 4, PT}, {0, 4, END}
};

CP Fdata[] = {
    {0, 0, PT}, {0, 10, PT}, {5, 10, STROKE},
    {0, 5, PT}, {4, 5, END}
};

CP Gdata[] = {
    {3, 5, PT}, {5, 5, PT}, {5, 1, PT}, {3, 0, PT},
    {1, 0, PT}, {0, 1, PT}, {0, 9, PT}, {1, 10, PT},
    {4, 10, PT}, {5, 9, PT}, {5, 7, STROKE},
    {5, 5, PT}, {5, -1, END}
};

CP Hdata[] = {
    {0, 0, PT}, {0, 10, STROKE}, {5, 0, PT}, {5, 10, STROKE},
    {0, 5, PT}, {5, 5, END}
};

CP Idata[] = {
    {0, 0, PT}, {4, 0, STROKE}, {0, 10, PT}, {4, 10, STROKE},
    {2, 0, PT}, {2, 10, END}
};

CP Jdata[] = {
    {2, 10, PT}, {6, 10, STROKE}, {4, 10, PT}, {4, 1, PT},
    {3, 0, PT}, {1, 0, PT}, {0, 1, PT}, {0, 4, END}
};

CP Kdata[] = {
    {0, 0, PT}, {0, 10, STROKE}, {5, 0, PT}, {0, 5, PT},
    {5, 10, END}
};

CP Ldata[] = {
    {5, 0, PT}, {0, 0, PT}, {0, 10, END}
};

CP Mdata[] = {
    {0, 0, PT}, {0, 10, PT}, {3, 6, PT},
    {6, 10, PT}, {6, 0, END}
};

CP Ndata[] = {
    {0, 0, PT}, {0, 10, PT}, {5, 0, PT},
    {5, 10, END}
};

CP Odata[] = {
    {0, 1, PT}, {1, 0, PT}, {4, 0, PT}, {5, 1, PT}, {5, 9, PT},
    {4, 10, PT}, {1, 10, PT}, {0, 9, PT}, {0, 1, END}
};

CP Pdata[] = {
    {0, 0, PT}, {0, 10, PT},  {4, 10, PT}, {5, 9, PT}, {5, 6, PT},
    {4, 5, PT}, {0, 5, END}
};

CP Qdata[] = {
    {0, 1, PT}, {1, 0, PT}, {4, 0, PT}, {5, 1, PT}, {5, 9, PT},
    {4, 10, PT}, {1, 10, PT}, {0, 9, PT}, {0, 1, STROKE},
    {0, 1, PT}, {2, 2, PT}, {4, 0, PT}, {5, -2, END}
};

CP Rdata[] = {
    {0, 0, PT}, {0, 10, PT},  {4, 10, PT}, {5, 9, PT}, {5, 6, PT},
    {4, 5, PT}, {0, 5, STROKE}, {3, 5, PT}, {5, 0, END}
};

CP Sdata[] = {
    {0, 1, PT}, {1, 0, PT}, {4, 0, PT}, {5, 1, PT}, {5, 4, PT},
    {4, 5, PT}, {1, 5, PT}, {0, 6, PT}, {0, 9, PT}, {1, 10, PT},
    {4, 10, PT}, {5, 9, END}
};

CP Tdata[] = {
    {0, 10, PT}, {6, 10, STROKE}, {3, 0, PT}, {3, 10, END}
};

CP Udata[] = {
    {0, 10, PT}, {0, 1, PT}, {1, 0, PT}, {4, 0, PT}, {5, 1, PT}, {5, 9, PT},
    {5, 10, END}
};

CP Vdata[] = {
    {0, 10, PT}, {3, 0, PT}, {6, 10, END}
};

CP Wdata[] = {
    {0, 10, PT}, {0, 0, PT}, {3, 4, PT}, {6, 0, PT}, {6, 10, END}
};

CP Xdata[] = {
    {0, 10, PT}, {5, 0, STROKE}, {0, 0, PT}, {6, 10, END}
};

CP Ydata[] = {
    {0, 10, PT}, {3, 5, PT}, {6, 10, STROKE}, {3, 5, PT}, {3, 0, END}
};

CP Zdata[] = {
    {0, 10, PT}, {5, 10, PT}, {0, 0, PT}, {5, 0, END}
};

CP ndata1[] = {
    {0, 0, PT}, {4, 0, STROKE}, {0, 7, PT}, {2, 10, STROKE},
    {2, 0, PT}, {2, 10, END}
};

CP onedata[] = {
    {0, 0, PT}, {4, 0, STROKE}, {0, 10, PT}, {4, 10, STROKE},
    {2, 0, PT}, {2, 10, END}
};

CP ndata2[] = {
    {0, 9, PT}, {1, 10, PT}, {4, 10, PT}, {5, 9, PT},
    {5, 6, PT}, {4, 5, PT}, {0, 0, PT},
    {6, 0, END}
};

CP ndata3[] = {
    {0, 10, PT}, {5, 10, PT}, {2, 6, PT}, {4, 6, PT},
    {5, 5, PT}, {5, 1, PT}, {4, 0, PT}, {1, 0, PT},
    {0, 1, END}
};

CP ndata4[] = {
    {4, 0, PT}, {4, 10, PT}, {0, 3, PT}, {6, 3, END}
};

CP ndata5[] = {
    {5, 10, PT}, {0, 10, PT}, {0, 6, PT}, {1, 7, PT},
    {4, 7, PT}, {5, 6, PT}, {5, 1, PT}, {4, 0, PT},
    {1, 0, PT}, {0, 1, END}
};

CP ndata6[] = {
    {0, 4, PT}, {1, 5, PT}, {4, 5, PT}, {5, 4, PT}, {5, 1, PT}, {3, 0, PT},
    {1, 0, PT}, {0, 1, PT}, {0, 9, PT}, {1, 10, PT},
    {4, 10, PT}, {5, 9, PT}, {5, 7, END},
};

CP ndata7[] = {
    {0, 9, PT}, {0, 10, PT}, {5, 10, PT}, {3, 5, PT}, {3, 0, END}
};

CP ndata8[] = {
    {0, 1, PT}, {1, 0, PT}, {4, 0, PT}, {5, 1, PT}, {5, 3, PT},
    {1, 6, PT}, {0, 8, PT}, {0, 9, PT}, {1, 10, PT}, {4, 10, PT},
    {5, 9, PT}, {5, 7, PT}, {1, 4, PT}, {0, 3, PT}, {0, 1, END}
};

CP ndata9[] = {
    {0, 1, PT}, {1, 0, PT}, {3, 0, PT}, {5, 2, PT}, {5, 9, PT},
    {4, 10, PT}, {1, 10, PT}, {0, 9, PT}, {0, 6, PT}, {1, 5, PT},
    {3, 5, PT}, {5, 6, END}
};

CP ndata0[] = {
    {0, 2, PT}, {3, 0, PT}, {6, 2, PT}, {6, 8, PT},
    {3, 10, PT}, {0, 8, PT}, {0, 2, END}
};

CP ndataplus[] = {
    {3, 2, PT}, {3, 8, STROKE}, {1, 5, PT}, {6, 5, END}
};

CP ndataminus[] = {
    {1, 5, PT}, {5, 5, END}
};

CP ndatadot[] = {
    {3, 1, PT}, {4, 1, PT}, {4, 0, PT}, {3, 0, PT}, {3, 1, END}
};


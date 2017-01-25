/* $Id: matrix.c,v 1.18.4.2 2000/11/05 21:24:01 brianp Exp $ */

/*
 * Mesa 3-D graphics library
 * Version:  3.4
 * 
 * Copyright (C) 1999-2000  Brian Paul   All Rights Reserved.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


/*
    This utility function calculates rotations based on Quaternions.
    It is based on the function:

      gl_rotation_matrix

    taken from the Mesa OpenGL library source file:

      /usr/local/Mesa-3.4/src/matrix.c

    and modified by San Le.

    I downloaded a new version of Mesa on 11/20/06, Mesa-6.4.2, and this
    function is now called:

      _math_matrix_rotate

    which is found in the file:

      /usr/local/Mesa-6.4.2/src/mesa/math/m_matrix.c


                Updated 8/13/07

    SLFFEA source file
    Version:  1.5

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "control.h"

static double Identity2[9] = {
   1.0, 0.0, 0.0,
   0.0, 1.0, 0.0,
   0.0, 0.0, 1.0
};

/*
 * Generate a 4x4 transformation matrix from glRotate parameters.
 */

int quaternion( double angle, double *rot_vec, double *M)
{
   /* This function contributed by Erich Boleyn (erich@uruk.org) */
   double mag, s, c, fdum;
   double x, y, z;
   double xx, yy, zz, xy, yz, zx, xs, ys, zs, one_c;

   fdum = angle * DEG2RAD;
   s = sin(fdum);
   c = cos(fdum);

   x = rot_vec[0]; y = rot_vec[1]; z = rot_vec[2];

   mag = sqrt( x*x + y*y + z*z );

   if (mag <= 1.0e-4) {
      /* generate an identity matrix and return */
      memcpy(M, Identity2, sizeof(double)*9);
      return 1;
   }

   x /= mag;
   y /= mag;
   z /= mag;

/* #define M(row,col)  m[col*4+row]*/

   /*
    *     Arbitrary axis rotation matrix.
    *
    *  This is composed of 5 matrices, Rz, Ry, T, Ry', Rz', multiplied
    *  like so:  Rz * Ry * T * Ry' * Rz'.  T is the final rotation
    *  (which is about the X-axis), and the two composite transforms
    *  Ry' * Rz' and Rz * Ry are (respectively) the rotations necessary
    *  from the arbitrary axis to the X-axis then back.  They are
    *  all elementary rotations.
    *
    *  Rz' is a rotation about the Z-axis, to bring the axis vector
    *  into the x-z plane.  Then Ry' is applied, rotating about the
    *  Y-axis to bring the axis vector parallel with the X-axis.  The
    *  rotation about the X-axis is then performed.  Ry and Rz are
    *  simply the respective inverse transforms to bring the arbitrary
    *  axis back to it's original orientation.  The first transforms
    *  Rz' and Ry' are considered inverses, since the data from the
    *  arbitrary axis gives you info on how to get to it, not how
    *  to get away from it, and an inverse must be applied.
    *
    *  The basic calculation used is to recognize that the arbitrary
    *  axis vector (x, y, z), since it is of unit length, actually
    *  represents the sines and cosines of the angles to rotate the
    *  X-axis to the same orientation, with theta being the angle about
    *  Z and phi the angle about Y (in the order described above)
    *  as follows:
    *
    *  cos ( theta ) = x / sqrt ( 1 - z^2 )
    *  sin ( theta ) = y / sqrt ( 1 - z^2 )
    *
    *  cos ( phi ) = sqrt ( 1 - z^2 )
    *  sin ( phi ) = z
    *
    *  Note that cos ( phi ) can further be inserted to the above
    *  formulas:
    *
    *  cos ( theta ) = x / cos ( phi )
    *  sin ( theta ) = y / sin ( phi )
    *
    *  ...etc.  Because of those relations and the standard trigonometric
    *  relations, it is pssible to reduce the transforms down to what
    *  is used below.  It may be that any primary axis chosen will give the
    *  same results (modulo a sign convention) using thie method.
    *
    *  Particularly nice is to notice that all divisions that might
    *  have caused trouble when parallel to certain planes or
    *  axis go away with care paid to reducing the expressions.
    *  After checking, it does perform correctly under all cases, since
    *  in all the cases of division where the denominator would have
    *  been zero, the numerator would have been zero as well, giving
    *  the expected result.
    */

   xx = x * x;
   yy = y * y;
   zz = z * z;
   xy = x * y;
   yz = y * z;
   zx = z * x;
   xs = x * s;
   ys = y * s;
   zs = z * s;
   one_c = 1.0F - c;

   *(M + 0) = (one_c * xx) + c;
   *(M + 1) = (one_c * xy) - zs;
   *(M + 2) = (one_c * zx) + ys;

   *(M + 3) = (one_c * xy) + zs;
   *(M + 4) = (one_c * yy) + c;
   *(M + 5) = (one_c * yz) - xs;

   *(M + 6) = (one_c * zx) - ys;
   *(M + 7) = (one_c * yz) + xs;
   *(M + 8) = (one_c * zz) + c;

   return 1;
}


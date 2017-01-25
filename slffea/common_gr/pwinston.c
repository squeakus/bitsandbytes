/*
    This program handles all the movement of the mesh based on
    mouse or GUI user input.  It is almost entirely(95%) taken from the
    work of Philip Winston in the program "walker"( specifically,
    the "walkviewer.c" module).  He has graciously
    and generously allowed me to use and modify it for these finite
    element graphics programs.
  
                                  San Le
  
   			Last Update 5/14/00
  
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

/***************************************************************/
/************************** SETTINGS ***************************/
/***************************************************************/

   /* Initial polar movement settings */
#define INIT_POLAR_AZ  0.0
#define INIT_POLAR_AY  0.0
#define INIT_POLAR_EL 30.0
#define INIT_DIST      3.0
#define INIT_AZ_SPIN   0.5
#define INIT_AY_SPIN   0.5
#define INIT_EL_SPIN   0.0

  /* Initial flying movement settings */
#define INIT_EX        0.0
#define INIT_EY       -2.0
#define INIT_EZ       -2.0
#define INIT_MOVE     0.01
#define MINMOVE      0.001

  /* Controls:  */

  /* What to multiply number of pixels mouse moved by to get rotation amount */
#define EL_SENS   0.5
#define AZ_SENS   0.5
#define AY_SENS   0.5

  /* What to multiply number of pixels mouse moved by for movement amounts */
#define DIST_SENS 0.01
#define E_SENS    0.01

  /* Minimum spin to allow in polar (lower forced to zero) */
#define MIN_AZSPIN 0.1
#define MIN_AYSPIN 0.1
#define MIN_ELSPIN 0.1

  /* Factors used in computing dAz dAy and dEl (which determine AySpin AzSpin, ElSpin) */
#define PREV_DAY 0.80
#define PREV_DAZ 0.80
#define PREV_DEL 0.80
#define CUR_DAY  0.20
#define CUR_DAZ  0.20
#define CUR_DEL  0.20

/***************************************************************/
/************************** GLOBALS ****************************/
/***************************************************************/

GLfloat Ex = INIT_EX,             /* flying parameters */
        Ey = INIT_EY,
        Ez = INIT_EZ,
        EyeMove = INIT_MOVE,

        EyeDist = INIT_DIST,      /* polar params */
        AzSpin  = INIT_AZ_SPIN,
        AySpin  = INIT_AY_SPIN,
        ElSpin  = INIT_EL_SPIN,

        EyeAz = INIT_POLAR_AZ,    /* used by both */
        EyeAy = INIT_POLAR_AY,    
        EyeEl = INIT_POLAR_EL;

int agvMoving;    /* Currently moving?  */

int downx, downy,   /* for tracking mouse position */
    lastx, lasty,
    downb = -1;     /* and button status */

GLfloat downDist, horzDist, vertDist, /* for saving state of things */
        downEl, downAy, downAz, 
        downEx, downEy, downEz,   /* when button is pressed */
        downEyeMove, vertEyeMove;

GLfloat dAy, dAz, dEl, lastAy, lastAz, lastEl;  /* to calculate spinning w/ polar motion */
int     AdjustingAzEl = 0, AdjustingAyEl = 0;

int AllowIdle, RedisplayWindow;
   /* If AllowIdle is 1 it means AGV will install its own idle which
    * will update the viewpoint as needed and send glutPostRedisplay() to the
    * window RedisplayWindow which was set in agvInit().  AllowIdle of 0
    * means AGV won't install an idle funciton, and something like
    * "if (agvMoving) agvMove()" should exist at the end of the running
    * idle function.
    */

/* Some <math.h> files do not define M_PI... */
#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif
#define TORAD(x) ((M_PI/180.0)*(x))
#define TODEG(x) ((180.0/M_PI)*(x))

/* For movement in FEM program */

extern double left_right, up_down, in_out, left_right0, up_down0, in_out0;
extern double step_sizex, step_sizey, step_sizez;
extern double xAngle, yAngle, zAngle;

/* Below is the routine for the mesh translation and rotation */

/***************************************************************/
/*******************    MOUSE HANDLING   ***********************/
/***********************lllll***********************************/

void agvHandleButton(int button, int state, int x, int y)
{
 if (state == GLUT_DOWN && downb == -1) {
    lastx = downx = x;
    lasty = downy = y;
    downb = button;

    switch (button) {
      case GLUT_LEFT_BUTTON:
        downDist = in_out;
        downEx = Ex;
        downEy = Ey;
        downEz = Ez;
        downEyeMove = EyeMove;
        EyeMove = 0;

        lastAy = downAy = -zAngle;
        AySpin = ElSpin = dAy = dEl = 0;
        AdjustingAyEl = 1;
        break;

      case GLUT_MIDDLE_BUTTON:
        horzDist = left_right;
        vertDist = -up_down;
        downEx = Ex;
        downEy = Ey;
        downEz = Ez;
        vertEyeMove = EyeMove;
        EyeMove = 0;
        break;

      case GLUT_RIGHT_BUTTON:
        lastEl = downEl = xAngle;
        lastAz = downAz = yAngle;
        AzSpin = ElSpin = dAz = dEl = 0;
        AdjustingAzEl = 1;
        /*MoveOn(1);*/
    }

  } else if (state == GLUT_UP && button == downb) {

    downb = -1;

    switch (button) {

      case GLUT_LEFT_BUTTON:
        EyeMove = downEyeMove;

        if (AySpin < MIN_AYSPIN && AySpin > -MIN_AYSPIN)
          AySpin = 0;

      case GLUT_MIDDLE_BUTTON:
        EyeMove = vertEyeMove;

      case GLUT_RIGHT_BUTTON:
          if (AzSpin < MIN_AZSPIN && AzSpin > -MIN_AZSPIN)
            AzSpin = 0;

          ElSpin = -dEl;
          if (ElSpin < MIN_ELSPIN && ElSpin > -MIN_ELSPIN)
            ElSpin = 0;
          AdjustingAzEl = 0;
          /*MoveOn(1);*/
          break;

      }
  }
}


 /*
  * change xAngle and yAngle and position when mouse is moved w/ button down
 */

void agvHandleMotion(int x, int y)
{
  int deltax = x - downx, deltay = y - downy;

  switch (downb) {
    case GLUT_LEFT_BUTTON:
        in_out = downDist + DIST_SENS*deltay;
        Ex = downEx - E_SENS*deltay*sin(TORAD(yAngle))*cos(TORAD(xAngle));
        Ey = downEy - E_SENS*deltay*sin(TORAD(xAngle));
        Ez = downEz + E_SENS*deltay*cos(TORAD(yAngle))*cos(TORAD(xAngle));

        zAngle  = -downAy - AY_SENS * deltax;
        dAy    = PREV_DAY*dAy + CUR_DAY*(lastAy - zAngle);
        lastAy = -zAngle;
      break;
    case GLUT_MIDDLE_BUTTON:
        up_down = - vertDist - DIST_SENS*deltay;
        left_right = horzDist + DIST_SENS*deltax;
        Ex = downEx - E_SENS*deltay*sin(TORAD(yAngle))*cos(TORAD(xAngle));
        Ey = downEy - E_SENS*deltay*sin(TORAD(xAngle));
        Ez = downEz + E_SENS*deltay*cos(TORAD(yAngle))*cos(TORAD(xAngle));
      break;
    case GLUT_RIGHT_BUTTON:
      xAngle  = downEl + EL_SENS * deltay;
      /*ConstrainEl();*/
      yAngle  = downAz + AZ_SENS * deltax;
      dAz    = PREV_DAZ*dAz + CUR_DAZ*(lastAz - yAngle);
      dEl    = PREV_DEL*dEl + CUR_DEL*(lastEl - xAngle);
      lastAz = yAngle;
      lastEl = xAngle;
      break;
  }
  glutPostRedisplay();
}


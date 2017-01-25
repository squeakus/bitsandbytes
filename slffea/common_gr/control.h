/*
    This is the include file "control.h" for the control panel
    part of the graphics program

                Updated 5/10/01

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define scaleFactor              0.10               /* Scale down text by this amount */
#define mesh_scaleFactor         0.01               /* Scale boxes in mesh window by this amount */
#define textHeight              10                  /* Height of text */
#define textHeightDiv           15                  /* Vertical distance between text */
#define textWidth               10                  /* width of text */
#define textMove_x0             12.5*textWidth      /* Positions text horizontally for scale bar */
#define boxTextMove_x0          17*textWidth        /* Positions box horrizontally */
#define scale_boxTextMove_x0    textWidth         /* Positions box horrizontally */
#define hscale_boxTextMove_x0   textWidth         /* Positions box horrizontally */
#define textDiv_xa0             5*textWidth         /* Division between toggle switch */
#define textDiv_xb0             11*textWidth        /* Horizontal distance between buttons */
#define boxHeight               3*textHeight        /* Horizontal distance between button and text */
#define boxdim                  30                  /* size of 1 scale box */
#define boxnumber                8                  /* number of boxes in scale */
#define left_indent             6*textHeight         /* Color Scale Box distance from left magin */
#define bottom_indent           2*textHeight - 2     /* Text distance from bottom */

#define rowdim                 45
#define control_width0        350    /* Initial width of control window */
#define control_height0       700    /* Initial height of control window */
#define mesh_width0           800    /* Initial width of mesh window */
#define mesh_height0          800    /* Initial height of mesh window */
#define scale_width0          200    /* Initial width of vertical scale window */
#define scale_height0         300    /* Initial height of vertical scale window */
#define hscale_width0         420    /* Initial width of horizontal scale window */
#define hscale_height0        150    /* Initial height of horizontal scale window */

#define BIG                 1.0e20
#define IBIG                1e7
#define SMALL               1.0e-20
#define SMALL2              1.0e-10
#define PI                  3.14159265358979323846
#define DEG2RAD             PI/180.0

#define LINUX               1
#define WINDOWS             0

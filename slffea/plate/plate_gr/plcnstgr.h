/* The plate needs some extra variables because there are more stress/strain nad
   moment/curvature quantities requiring a larger control panel.

        Updated 9/24/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define plrowdim               54
#define plcontrol_width0      350    /* Initial width of control window */
#define plcontrol_height0     800    /* Initial height of control window */


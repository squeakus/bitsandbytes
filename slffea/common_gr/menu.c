/*
    This program draws the drag down menus.  It works with the FEM code.
  
	                Last Update 8/12/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
 */

#if WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#if BRICK1
#include "../brick/brick/brstruct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if BRICK2
#include "../brick/brick2/br2struct.h"
#include "../brick/brick_gr/brstrcgr.h"
#endif
#if QUAD1
#include "../quad/quad/qdstruct.h"
#include "../quad/quad_gr/qdstrcgr.h"
#endif
#if WEDGE1
#include "../wedge/wedge/westruct.h"
#include "../wedge/wedge_gr/westrcgr.h"
#endif


#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern int nmat, numnp, numel, dof;
extern double *coord, *coord0;
extern double *U;
extern int *connecter;
extern BOUND bc;
extern double *force;
extern SDIM *stress_node;
extern SDIM *strain_node;
extern XYZF *force_vec, *force_vec0;
extern ISTRESS *stress_color;
extern ISTRAIN *strain_color;
extern int *U_color;

extern int input_flag, post_flag, color_choice,
    choice, matl_choice, node_choice, ele_choice;
extern int input_color_flag;
extern int Solid_flag, Perspective_flag, Render_flag,
    AppliedDisp_flag, AppliedForce_flag,
    Material_flag, Node_flag, Element_flag, Axes_flag;
extern int Before_flag, After_flag,
    Both_flag, Amplify_flag;
extern double amplify_factor, amplify_step, amplify_step0;

#if BRICK2
extern double *heat_el, *heat_node, *T, *Q;
extern int *T_color, *Q_color;
#endif

#if BRICK1 || WEDGE1
int set(BOUND , int *, double *, XYZF *, SDIM *, ISTRAIN *,
	SDIM *, ISTRESS *, double *, int *);
#endif
#if BRICK2
int set(BOUND , int *, double *, XYZF *, double *, int *, SDIM *,
	ISTRAIN *, SDIM *, ISTRESS *, double *, int *, double *, int *);
#endif
#if QUAD1
int qdset(BOUND , int *, double *, XYZF *, SDIM *, ISTRAIN *,
	SDIM *, ISTRESS *, double *, int *);
#endif

#if BRICK1 || WEDGE1 || BRICK2
void ReGetparameter( void);
#endif

#if QUAD1
void ReGetparameter2( void);
#endif

void MenuSelect(int value)
{
	int check;

	switch (value) {
	case 1:
	    color_choice = 31;
	    input_color_flag = 0;
	    AppliedForce_flag = 0;
	    AppliedDisp_flag = 0;
	    Element_flag = 0;
	    Material_flag = 0;
	    Node_flag = 1;

	    printf("\n What is the desired node number?\n");
	    scanf("%d", &node_choice);
	    if ( node_choice > numnp - 1 )
	    {
		node_choice = 0;
	    }
	    break;
	case 2:
	    color_choice = 32;
	    input_color_flag = 0;
	    AppliedForce_flag = 0;
	    AppliedDisp_flag = 0;
	    Element_flag = 1;
	    Material_flag = 0;
	    Node_flag = 0;

	    printf("\n What is the desired element number?\n");
	    scanf("%d", &ele_choice);
	    if ( ele_choice > numel - 1 )
	    {
		ele_choice = 0;
	    }
	    Solid_flag = 1;
	    break;
	case 3:
	    color_choice = 30;
	    input_color_flag = 0;
	    AppliedForce_flag = 0;
	    AppliedDisp_flag = 0;
	    Element_flag = 0;
	    Material_flag = 1;
	    Node_flag = 0;

	    printf("\n What is the desired material number?\n");
	    scanf("%d", &matl_choice);
	    if ( matl_choice > nmat - 1 )
	    {
		matl_choice = 0;
	    }
	    Solid_flag = 1;
	    break;
	case 4:
#if BRICK1 || WEDGE1 || BRICK2
	    ReGetparameter();
#endif

#if BRICK1 || WEDGE1
	    check = set( bc, connecter, force, force_vec0, strain_node,
		strain_color, stress_node, stress_color, U, U_color);
	    if(!check) printf( " Problems with set \n");
#endif
#if BRICK2
	    check = set( bc, connecter, force, force_vec0, Q, Q_color,
		strain_node, strain_color, stress_node, stress_color, T,
		T_color, U, U_color);
#endif
#if QUAD1
	    check = qdset( bc, connecter, force, force_vec0, strain_node,
		strain_color, stress_node, stress_color, U, U_color);
	    if(!check) printf( " Problems with set \n");

	    ReGetparameter2();
#endif
	    break;
	case 5:
	    exit(0);
	    break;
	}

	glutPostRedisplay();

}

void Menu(void)
{
	glutCreateMenu(MenuSelect);
	glutAddMenuEntry("Jump node", 1);
	glutAddMenuEntry("Jump ele", 2);
	glutAddMenuEntry("Jump Matl", 3);
	glutAddMenuEntry("Re-Param", 4);
	glutAddMenuEntry("Quit", 5);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

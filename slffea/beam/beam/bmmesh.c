/*
   This program plots the mesh with the various
   forms of viewing including moment, curve, stress,
   strain, displacement materials, etc.  It works
   with a beam FEM code.

                     Last Update 9/23/06

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
#include "../beam/bmconst.h"
#include "../beam/bmstruct.h"
#include "bmstrcgr.h"
#include "../../common_gr/control.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* FEA globals */

extern double *coord, *coord0;
extern int *connecter;
extern int nmat, numnp, numel;
extern GLfloat MeshColor[boxnumber+5][4];
extern GLfloat wire_color[4], black[4], green[4], yellow[4];
extern GLfloat RenderColor[4];
extern IMOMENT *moment_color;
extern ICURVATURE *curve_color;
extern ISTRESS *stress_color;
extern ISTRAIN *strain_color;
extern int *U_color, *el_matl_color;
extern int color_choice, input_flag, post_flag;
extern int input_color_flag;
extern int Perspective_flag, Render_flag, AppliedDisp_flag,
	AppliedForce_flag, Material_flag, Node_flag, Element_flag, Axes_flag,
	Transparent_flag, CrossSection_flag;
extern int Before_flag, After_flag, Both_flag, Amplify_flag;
extern int stress_flag, strain_flag, disp_flag;
extern int matl_choice, node_choice, ele_choice;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out;

void bmmeshdraw(void)
{
	int i, i2, j, k, dof_el[neqel], ii, check, counter,
		node0, node1;
	int l,m,n;
	int c0,c1;
	int matl_number, node_number;
	int After_gr_flag = 0, Before_gr_flag = 0,
		After_element_draw_flag = 1, Before_element_draw_flag = 1;
	double coord_el[npel*3], coord0_el[npel*3], fpointx, fpointy, fpointz;
	GLfloat d1[3], d2[3];

	if(post_flag + After_flag > 1) After_gr_flag = 1;
	else After_flag = 0;
	if(input_flag + Before_flag > 1) Before_gr_flag = 1;
	else Before_flag = 0;

	*(wire_color + 2) = 0.0;

	MeshColor[0][3] = 1.0;
	MeshColor[1][3] = 1.0;
	MeshColor[2][3] = 1.0;
	MeshColor[3][3] = 1.0;
	MeshColor[4][3] = 1.0;
	MeshColor[5][3] = 1.0;
	MeshColor[6][3] = 1.0;
	MeshColor[7][3] = 1.0;

	if(Transparent_flag)
	{
		MeshColor[0][3] = 0.075;
		MeshColor[1][3] = 0.075;
		MeshColor[2][3] = 0.075;
		MeshColor[3][3] = 0.075;
		MeshColor[4][3] = 0.075;
		MeshColor[5][3] = 0.075;
		MeshColor[6][3] = 0.075;
		MeshColor[7][3] = 0.075;
	}
	if(color_choice == 30 || color_choice == 32) MeshColor[7][3] = 1.0;

	for( k = 0; k < numel; ++k )
	{
		After_element_draw_flag = 1;
		Before_element_draw_flag = 1;

/* Calculate element degrees of freedom */

		node0 = *(connecter+k*npel);
		node1 = *(connecter+k*npel+1);

		*(dof_el) = ndof*node0;
		*(dof_el+1) = ndof*node0+1;
		*(dof_el+2) = ndof*node0+2;
		*(dof_el+3) = ndof*node0+3;
		*(dof_el+4) = ndof*node0+4;
		*(dof_el+5) = ndof*node0+5;

		*(dof_el+6) = ndof*node1;
		*(dof_el+7) = ndof*node1+1;
		*(dof_el+8) = ndof*node1+2;
		*(dof_el+9) = ndof*node1+3;
		*(dof_el+10) = ndof*node1+4;
		*(dof_el+11) = ndof*node1+5;

/* Calculate local deformed coordinates */

		if( post_flag )
		{
		    *(coord_el)=*(coord+nsd*node0);
		    *(coord_el+1)=*(coord+nsd*node0+1);
		    *(coord_el+2)=*(coord+nsd*node0+2);

		    *(coord_el+3)=*(coord+nsd*node1);
		    *(coord_el+4)=*(coord+nsd*node1+1);
		    *(coord_el+5)=*(coord+nsd*node1+2);

		    if( *(coord_el) > cross_sec_left_right)
			After_element_draw_flag = 0;
		    if( *(coord_el+1) > cross_sec_up_down)
			After_element_draw_flag = 0;
		    if( *(coord_el+2) > cross_sec_in_out)
			After_element_draw_flag = 0;
		    if( *(coord_el+3) > cross_sec_left_right)
			After_element_draw_flag = 0;
		    if( *(coord_el+4) > cross_sec_up_down)
			After_element_draw_flag = 0;
		    if( *(coord_el+5) > cross_sec_in_out)
			After_element_draw_flag = 0;

		}

/* Calculate local undeformed coordinates */

		if( input_flag )
		{
		    *(coord0_el)=*(coord0+nsd*node0);
		    *(coord0_el+1)=*(coord0+nsd*node0+1);
		    *(coord0_el+2)=*(coord0+nsd*node0+2);

		    *(coord0_el+3)=*(coord0+nsd*node1);
		    *(coord0_el+4)=*(coord0+nsd*node1+1);
		    *(coord0_el+5)=*(coord0+nsd*node1+2);

		    if( *(coord0_el) > cross_sec_left_right)
			Before_element_draw_flag = 0;
		    if( *(coord0_el+1) > cross_sec_up_down)
			Before_element_draw_flag = 0;
		    if( *(coord0_el+2) > cross_sec_in_out)
			Before_element_draw_flag = 0;
		    if( *(coord0_el+3) > cross_sec_left_right)
			Before_element_draw_flag = 0;
		    if( *(coord0_el+4) > cross_sec_up_down)
			Before_element_draw_flag = 0;
		    if( *(coord0_el+5) > cross_sec_in_out)
			Before_element_draw_flag = 0;
		}

		if(!CrossSection_flag)
		{
		    After_element_draw_flag = 1;
		    Before_element_draw_flag = 1;
		}

		/*printf( "%9.5f %9.5f %9.5f \n",*(coord_el+3*j),
			*(coord_el+3*j+1),*(coord_el+3*j+2));*/


/* Calculate element material number */

		matl_number = *(el_matl_color + k);

		switch (color_choice) {
		    case 41:
			c0 = curve_color[k].pt[0].xx;
			c1 = curve_color[k].pt[1].xx;
		    break;
		    case 42:
			c0 = curve_color[k].pt[0].yy;
			c1 = curve_color[k].pt[1].yy;
		    break;
		    case 43:
			c0 = curve_color[k].pt[0].zz;
			c1 = curve_color[k].pt[1].zz;
		    break;
		    case 1:
			c0 = strain_color[k].pt[0].xx;
			c1 = strain_color[k].pt[1].xx;
		    break;
		    case 4:
			c0 = strain_color[k].pt[0].xy;
			c1 = strain_color[k].pt[1].xy;
		    break;
		    case 5:
			c0 = strain_color[k].pt[0].zx;
			c1 = strain_color[k].pt[1].zx;
		    break;
		    case 50:
			c0 = moment_color[k].pt[0].xx;
			c1 = moment_color[k].pt[1].xx;
		    break;
		    case 51:
			c0 = moment_color[k].pt[0].yy;
			c1 = moment_color[k].pt[1].yy;
		    break;
		    case 52:
			c0 = moment_color[k].pt[0].zz;
			c1 = moment_color[k].pt[1].zz;
		    break;
		    case 10:
			c0 = stress_color[k].pt[0].xx;
			c1 = stress_color[k].pt[1].xx;
		    break;
		    case 13:
			c0 = stress_color[k].pt[0].xy;
			c1 = stress_color[k].pt[1].xy;
		    break;
		    case 14:
			c0 = stress_color[k].pt[0].zx;
			c1 = stress_color[k].pt[1].zx;
		    break;
		    case 19:
			c0 = *(U_color + *(dof_el + ndof*0));
			c1 = *(U_color + *(dof_el + ndof*1));
		    break;
		    case 20:
			c0 = *(U_color + *(dof_el + ndof*0 + 1));
			c1 = *(U_color + *(dof_el + ndof*1 + 1));
		    break;
		    case 21:
			c0 = *(U_color + *(dof_el + ndof*0 + 2));
			c1 = *(U_color + *(dof_el + ndof*1 + 2));
		    break;
		    case 22:
			c0 = *(U_color + *(dof_el + ndof*0 + 3));
			c1 = *(U_color + *(dof_el + ndof*1 + 3));
		    break;
		    case 23:
			c0 = *(U_color + *(dof_el + ndof*0 + 4));
			c1 = *(U_color + *(dof_el + ndof*1 + 4));
		    break;
		    case 24:
			c0 = *(U_color + *(dof_el + ndof*0 + 5));
			c1 = *(U_color + *(dof_el + ndof*1 + 5));
		    break;
		    case 30:
			c0 = 0;
			c1 = 0;
			if( matl_choice == matl_number )
			{
				c0 = 7;
				c1 = 7;
			}
		    break;
		    case 31:
			c0 = 0;
			c1 = 0;
		    break;
		    case 32:
			c0 = 0;
			c1 = 0;
			if( ele_choice == k )
			{
				c0 = 7;
				c1 = 7;
			}
		    break;
		}

/* Draw the mesh after deformation */

		if( After_gr_flag && After_element_draw_flag )
		{
			glBegin(GL_LINES);
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord_el));
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord_el+3));
			glEnd();
		}

		if( input_color_flag )
		{
		     c0 = 8;
		     c1 = 8;
		}

/* Draw the mesh before deformation */

		if( Before_gr_flag && Before_element_draw_flag )
		{
			glBegin(GL_LINES);
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord0_el));
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord0_el+3));
			glEnd();
		}
	}
/* This draws the Node ID node */
	if (color_choice == 31)
	{
	    glPointSize(8);
	    node_number=node_choice;
	    if( After_gr_flag )
	    {
		fpointx = *(coord+nsd*node_number);
		fpointy = *(coord+nsd*node_number+1);
		fpointz = *(coord+nsd*node_number+2);
		glBegin(GL_POINTS);
		    glColor4fv(yellow);
		    glVertex3f(fpointx, fpointy, fpointz);
		glEnd();
	    }
	    if( Before_gr_flag )
	    {
		fpointx = *(coord0+nsd*node_number);
		fpointy = *(coord0+nsd*node_number+1);
		fpointz = *(coord0+nsd*node_number+2);
		glBegin(GL_POINTS);
		    glColor4fv(yellow);
		    glVertex3f(fpointx, fpointy, fpointz);
		glEnd();
	    }
	}
	/*return 1;*/
}


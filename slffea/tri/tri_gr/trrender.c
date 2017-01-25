/*
    This program plots the mesh with the various
    forms of viewing including stress, strain, displacement
    materials, etc.  It works with a triangle FEM code.

                  Last Update 1/22/06

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
#include "../tri/trconst.h"
#include "../tri/trstruct.h"
#include "trstrcgr.h"
#include "../../common_gr/control.h"

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

/* FEA globals */

extern double *coord, *coord0;
extern NORM *norm, *norm0;
extern int *connecter;
extern int nmat, numnp, numel;
extern GLfloat MeshColor[boxnumber+5][4];
extern GLfloat wire_color[4], black[4], green[4], yellow[4];
extern GLfloat RenderColor[4];
extern ISDIM *stress_color;
extern ISDIM *strain_color;
extern int *U_color, *el_matl_color;
extern int color_choice, input_flag, post_flag;
extern int input_color_flag;
extern int Solid_flag, Perspective_flag, Render_flag, AppliedDisp_flag,
	AppliedForce_flag, Material_flag, Node_flag, Element_flag, Axes_flag,
	Transparent_flag, Outline_flag, CrossSection_flag;
extern int Before_flag, After_flag, Both_flag, Amplify_flag;
extern int stress_flag, strain_flag, stress_strain, disp_flag;
extern int matl_choice, node_choice, ele_choice;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out;

void trrender(void)
{
	int i, i2, j, k, dof_el[neqel], sdof_el[npel*nsd], ii, check, counter,
		node, node0, node1, node2;
	int l,m,n;
	int c0,c1,c2;
	int matl_number, node_number;
	int After_gr_flag = 0, Before_gr_flag = 0,
		After_element_draw_flag = 1, Before_element_draw_flag = 1;
	double coord_el[npel*3], coord0_el[npel*3], fpointx, fpointy, fpointz;
	GLfloat d1[3], d2[3], norm_temp[3];

	if(post_flag + After_flag > 1) After_gr_flag = 1;
	else After_flag = 0;
	if(input_flag + Before_flag > 1) Before_gr_flag = 1;
	else Before_flag = 0;

	*(wire_color + 2) = 0.0;
	if(!Solid_flag) *(wire_color + 2) = 1.0;

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
		MeshColor[0][3] = 0.175;
		MeshColor[1][3] = 0.175;
		MeshColor[2][3] = 0.175;
		MeshColor[3][3] = 0.175;
		MeshColor[4][3] = 0.175;
		MeshColor[5][3] = 0.175;
		MeshColor[6][3] = 0.175;
		MeshColor[7][3] = 0.175;
	}
	if(color_choice == 30 || color_choice == 32) MeshColor[7][3] = 1.0;

	for( k = 0; k < numel; ++k )
	{
		After_element_draw_flag = 1;
		Before_element_draw_flag = 1;

		for( j = 0; j < npel; ++j )
		{

/* Calculate element degrees of freedom */

		    node = *(connecter+npel*k+j);
		    *(sdof_el+nsd*j) = nsd*node;
		    *(sdof_el+nsd*j+1) = nsd*node+1;
		    *(sdof_el+nsd*j+2) = nsd*node+2;

		    *(dof_el+ndof*j) = ndof*node;
		    *(dof_el+ndof*j+1) = ndof*node+1;
		    *(dof_el+ndof*j+2) = ndof*node+2;

/* Calculate local deformed coordinates */

		    if( post_flag )
		    {
			*(coord_el+3*j)=*(coord+*(sdof_el+nsd*j));
			*(coord_el+3*j+1)=*(coord+*(sdof_el+nsd*j+1));
			*(coord_el+3*j+2)=*(coord+*(sdof_el+nsd*j+2));
			if( *(coord_el+3*j) > cross_sec_left_right)
				After_element_draw_flag = 0;
			if( *(coord_el+3*j + 1) > cross_sec_up_down)
				After_element_draw_flag = 0;
			if( *(coord_el+3*j + 2) > cross_sec_in_out)
				After_element_draw_flag = 0;
		    }

/* Calculate local undeformed coordinates */

		    if( input_flag )
		    {
			*(coord0_el+3*j)=*(coord0+*(sdof_el+nsd*j));
			*(coord0_el+3*j+1)=*(coord0+*(sdof_el+nsd*j+1));
			*(coord0_el+3*j+2)=*(coord0+*(sdof_el+nsd*j+2));
			if( *(coord0_el+3*j) > cross_sec_left_right)
				Before_element_draw_flag = 0;
			if( *(coord0_el+3*j + 1) > cross_sec_up_down)
				Before_element_draw_flag = 0;
			if( *(coord0_el+3*j + 2) > cross_sec_in_out)
				Before_element_draw_flag = 0;
		    }

			/*printf( "%9.5f %9.5f %9.5f \n",*(coord_el+3*j),
				*(coord_el+3*j+1),*(coord_el+3*j+2));*/
		}
		if(!CrossSection_flag)
		{
		    After_element_draw_flag = 1;
		    Before_element_draw_flag = 1;
		}

/* Calculate element material number */

		matl_number = *(el_matl_color + k);

		node0 = *(connecter+npel*k);
		node1 = *(connecter+npel*k + 1);
		node2 = *(connecter+npel*k + 2);

		switch (color_choice) {
		    case 1:
			c0 = strain_color[node0].xx;
			c1 = strain_color[node1].xx;
			c2 = strain_color[node2].xx;
		    break;
		    case 2:
			c0 = strain_color[node0].yy;
			c1 = strain_color[node1].yy;
			c2 = strain_color[node2].yy;
		    break;
		    case 4:
			c0 = strain_color[node0].xy;
			c1 = strain_color[node1].xy;
			c2 = strain_color[node2].xy;
		    break;
		    case 7:
			c0 = strain_color[node0].I;
			c1 = strain_color[node1].I;
			c2 = strain_color[node2].I;
		    break;
		    case 8:
			c0 = strain_color[node0].II;
			c1 = strain_color[node1].II;
			c2 = strain_color[node2].II;
		    break;
		    case 10:
			c0 = stress_color[node0].xx;
			c1 = stress_color[node1].xx;
			c2 = stress_color[node2].xx;
		    break;
		    case 11:
			c0 = stress_color[node0].yy;
			c1 = stress_color[node1].yy;
			c2 = stress_color[node2].yy;
		    break;
		    case 13:
			c0 = stress_color[node0].xy;
			c1 = stress_color[node1].xy;
			c2 = stress_color[node2].xy;
		    break;
		    case 16:
			c0 = stress_color[node0].I;
			c1 = stress_color[node1].I;
			c2 = stress_color[node2].I;
		    break;
		    case 17:
			c0 = stress_color[node0].II;
			c1 = stress_color[node1].II;
			c2 = stress_color[node2].II;
		    break;
		    case 19:
			c0 = *(U_color + *(dof_el + ndof*0));
			c1 = *(U_color + *(dof_el + ndof*1));
			c2 = *(U_color + *(dof_el + ndof*2));
		    break;
		    case 20:
			c0 = *(U_color + *(dof_el + ndof*0 + 1));
			c1 = *(U_color + *(dof_el + ndof*1 + 1));
			c2 = *(U_color + *(dof_el + ndof*2 + 1));
		    break;
		    case 21:
			c0 = *(U_color + *(dof_el + ndof*0 + 2));
			c1 = *(U_color + *(dof_el + ndof*1 + 2));
			c2 = *(U_color + *(dof_el + ndof*2 + 2));
		    break;
		    case 30:
			c0 = 0;
			c1 = 0;
			c2 = 0;
			if( matl_choice == matl_number )
			{
				c0 = 7;
				c1 = 7;
				c2 = 7;
			}
		    break;
		    case 31:
			c0 = 0;
			c1 = 0;
			c2 = 0;
		    break;
		    case 32:
			c0 = 0;
			c1 = 0;
			c2 = 0;
			if( ele_choice == k )
			{
				c0 = 7;
				c1 = 7;
				c2 = 7;
			}
		    break;
		}

/* Draw the mesh after deformation */

		if( After_gr_flag && After_element_draw_flag )
		{
		   if( Solid_flag )
		   {

/* Triangle face 0 */

			*(norm_temp) = norm[k].face[0].x;
			*(norm_temp+1) = norm[k].face[0].y;
			*(norm_temp+2) = norm[k].face[0].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c1]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c1]);
				glVertex3dv((coord_el+3));
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c0]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c0]);
				glVertex3dv((coord_el));
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c2]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c2]);
				glVertex3dv((coord_el+6));
			glEnd();
		   }
   
/* Draw the wire frame around the mesh */
   
		   if( Outline_flag )
		   {
			glMaterialfv(GL_FRONT, GL_DIFFUSE, wire_color);
			glMaterialfv(GL_FRONT, GL_AMBIENT, wire_color);
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord_el+3));
				glVertex3dv((coord_el));
				glVertex3dv((coord_el+6));
			glEnd();
		   }
		}

		if( input_color_flag )
		{
		     c0 = 8;
		     c1 = 8;
		     c2 = 8;
		}

/* Draw the mesh before deformation */

		if( Before_gr_flag && Before_element_draw_flag )
		{
		   if( Solid_flag )
		   {

/* Triangle face 0 */

			*(norm_temp) = norm0[k].face[0].x;
			*(norm_temp+1) = norm0[k].face[0].y;
			*(norm_temp+2) = norm0[k].face[0].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c1]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c1]);
				glVertex3dv((coord0_el+3));
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c0]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c0]);
				glVertex3dv((coord0_el));
				glMaterialfv(GL_FRONT, GL_DIFFUSE, MeshColor[c2]);
				glMaterialfv(GL_FRONT, GL_AMBIENT, MeshColor[c2]);
				glVertex3dv((coord0_el+6));
			glEnd();
		   }
   
/* Draw the wire frame around the mesh */
   
		   if( Outline_flag )
		   {
			glMaterialfv(GL_FRONT, GL_DIFFUSE, wire_color);
			glMaterialfv(GL_FRONT, GL_AMBIENT, wire_color);
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord0_el+3));
				glVertex3dv((coord0_el));
				glVertex3dv((coord0_el+6));
			glEnd();
		   }
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
		    glMaterialfv(GL_FRONT, GL_DIFFUSE, yellow);
		    glMaterialfv(GL_FRONT, GL_AMBIENT, yellow);
		    glVertex3f(fpointx, fpointy, fpointz);
		glEnd();
	    }
	    if( Before_gr_flag )
	    {
		fpointx = *(coord0+nsd*node_number);
		fpointy = *(coord0+nsd*node_number+1);
		fpointz = *(coord0+nsd*node_number+2);
		glBegin(GL_POINTS);
		    glMaterialfv(GL_FRONT, GL_DIFFUSE, yellow);
		    glMaterialfv(GL_FRONT, GL_AMBIENT, yellow);
		    glVertex3f(fpointx, fpointy, fpointz);
		glEnd();
	    }
	}
	/*return 1;*/
}


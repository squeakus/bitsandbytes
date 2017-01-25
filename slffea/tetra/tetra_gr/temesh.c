/*
    This program plots the mesh with the various
    forms of viewing including stress, strain, displacement
    materials, etc.  It works with a tetrahedral FEM code.

                  Last Update 1/20/06

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
#include "../tetra/teconst.h"
#if TETRA1
#include "../tetra/testruct.h"
#endif
#if TETRA2
#include "../tetra2/te2struct.h"
#endif
#include "testrcgr.h"
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
#if TETRA2
extern int *T_color, *Q_color;
#endif
extern int color_choice, input_flag, post_flag;
extern int input_color_flag;
extern int Solid_flag, Perspective_flag, Render_flag, AppliedDisp_flag,
	AppliedForce_flag, Material_flag, Node_flag, Element_flag, Axes_flag,
	Outline_flag, Transparent_flag, CrossSection_flag;
extern int Before_flag, After_flag, Both_flag, Amplify_flag;
extern int stress_flag, strain_flag, stress_strain, disp_flag;
extern int matl_choice, node_choice, ele_choice;
extern double cross_sec_left_right, cross_sec_up_down, cross_sec_in_out;

void temeshdraw(void)
{
	int i, i2, j, k, dof_el[neqel], Tdof_el[Tneqel], sdof_el[npel*nsd],
		ii, check, counter,
		node, node0, node1, node2, node3;
	int l,m,n;
	int c0,c1,c2,c3;
	int matl_number, node_number;
	int After_gr_flag = 0, Before_gr_flag = 0,
		After_element_draw_flag = 1, Before_element_draw_flag = 1;
	double coord_el[npel*3], coord0_el[npel*3],
		fpointx, fpointy, fpointz;
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
#if TETRA2
		    *(Tdof_el+Tndof*j) = Tndof*node;
#endif

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
		node3 = *(connecter+npel*k + 3);

		switch (color_choice) {
		    case 1:
			c0 = strain_color[node0].xx;
			c1 = strain_color[node1].xx;
			c2 = strain_color[node2].xx;
			c3 = strain_color[node3].xx;
		    break;
		    case 2:
			c0 = strain_color[node0].yy;
			c1 = strain_color[node1].yy;
			c2 = strain_color[node2].yy;
			c3 = strain_color[node3].yy;
		    break;
		    case 3:
			c0 = strain_color[node0].zz;
			c1 = strain_color[node1].zz;
			c2 = strain_color[node2].zz;
			c3 = strain_color[node3].zz;
		    break;
		    case 4:
			c0 = strain_color[node0].xy;
			c1 = strain_color[node1].xy;
			c2 = strain_color[node2].xy;
			c3 = strain_color[node3].xy;
		    break;
		    case 5:
			c0 = strain_color[node0].zx;
			c1 = strain_color[node1].zx;
			c2 = strain_color[node2].zx;
			c3 = strain_color[node3].zx;
		    break;
		    case 6:
			c0 = strain_color[node0].yz;
			c1 = strain_color[node1].yz;
			c2 = strain_color[node2].yz;
			c3 = strain_color[node3].yz;
		    break;
		    case 7:
			c0 = strain_color[node0].I;
			c1 = strain_color[node1].I;
			c2 = strain_color[node2].I;
			c3 = strain_color[node3].I;
		    break;
		    case 8:
			c0 = strain_color[node0].II;
			c1 = strain_color[node1].II;
			c2 = strain_color[node2].II;
			c3 = strain_color[node3].II;
		    break;
		    case 9:
			c0 = strain_color[node0].III;
			c1 = strain_color[node1].III;
			c2 = strain_color[node2].III;
			c3 = strain_color[node3].III;
		    break;
		    case 10:
			c0 = stress_color[node0].xx;
			c1 = stress_color[node1].xx;
			c2 = stress_color[node2].xx;
			c3 = stress_color[node3].xx;
		    break;
		    case 11:
			c0 = stress_color[node0].yy;
			c1 = stress_color[node1].yy;
			c2 = stress_color[node2].yy;
			c3 = stress_color[node3].yy;
		    break;
		    case 12:
			c0 = stress_color[node0].zz;
			c1 = stress_color[node1].zz;
			c2 = stress_color[node2].zz;
			c3 = stress_color[node3].zz;
		    break;
		    case 13:
			c0 = stress_color[node0].xy;
			c1 = stress_color[node1].xy;
			c2 = stress_color[node2].xy;
			c3 = stress_color[node3].xy;
		    break;
		    case 14:
			c0 = stress_color[node0].zx;
			c1 = stress_color[node1].zx;
			c2 = stress_color[node2].zx;
			c3 = stress_color[node3].zx;
		    break;
		    case 15:
			c0 = stress_color[node0].yz;
			c1 = stress_color[node1].yz;
			c2 = stress_color[node2].yz;
			c3 = stress_color[node3].yz;
		    break;
		    case 16:
			c0 = stress_color[node0].I;
			c1 = stress_color[node1].I;
			c2 = stress_color[node2].I;
			c3 = stress_color[node3].I;
		    break;
		    case 17:
			c0 = stress_color[node0].II;
			c1 = stress_color[node1].II;
			c2 = stress_color[node2].II;
			c3 = stress_color[node3].II;
		    break;
		    case 18:
			c0 = stress_color[node0].III;
			c1 = stress_color[node1].III;
			c2 = stress_color[node2].III;
			c3 = stress_color[node3].III;
		    break;
		    case 19:
			c0 = *(U_color + *(dof_el + ndof*0));
			c1 = *(U_color + *(dof_el + ndof*1));
			c2 = *(U_color + *(dof_el + ndof*2));
			c3 = *(U_color + *(dof_el + ndof*3));
		    break;
		    case 20:
			c0 = *(U_color + *(dof_el + ndof*0 + 1));
			c1 = *(U_color + *(dof_el + ndof*1 + 1));
			c2 = *(U_color + *(dof_el + ndof*2 + 1));
			c3 = *(U_color + *(dof_el + ndof*3 + 1));
		    break;
		    case 21:
			c0 = *(U_color + *(dof_el + ndof*0 + 2));
			c1 = *(U_color + *(dof_el + ndof*1 + 2));
			c2 = *(U_color + *(dof_el + ndof*2 + 2));
			c3 = *(U_color + *(dof_el + ndof*3 + 2));
		    break;
#if TETRA2
		    case 22:
			c0 = *(T_color + *(Tdof_el + Tndof*0));
			c1 = *(T_color + *(Tdof_el + Tndof*1));
			c2 = *(T_color + *(Tdof_el + Tndof*2));
			c3 = *(T_color + *(Tdof_el + Tndof*3));
		    break;
		    case 23:
			c0 = *(Q_color + *(Tdof_el + Tndof*0));
			c1 = *(Q_color + *(Tdof_el + Tndof*1));
			c2 = *(Q_color + *(Tdof_el + Tndof*2));
			c3 = *(Q_color + *(Tdof_el + Tndof*3));
		    break;
#endif
		    case 30:
			c0 = 0;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			if( matl_choice == matl_number )
			{
				c0 = 7;
				c1 = 7;
				c2 = 7;
				c3 = 7;
			}
		    break;
		    case 31:
			c0 = 0;
			c1 = 0;
			c2 = 0;
			c3 = 0;
		    break;
		    case 32:
			c0 = 0;
			c1 = 0;
			c2 = 0;
			c3 = 0;
			if( ele_choice == k )
			{
				c0 = 7;
				c1 = 7;
				c2 = 7;
				c3 = 7;
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
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord_el+3));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord_el));
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord_el+6));
			glEnd();

/* Triangle face 1 */

			*(norm_temp) = norm[k].face[1].x;
			*(norm_temp+1) = norm[k].face[1].y;
			*(norm_temp+2) = norm[k].face[1].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord_el+3));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord_el+9));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord_el));
			glEnd();

/* Triangle face 2 */

			*(norm_temp) = norm[k].face[2].x;
			*(norm_temp+1) = norm[k].face[2].y;
			*(norm_temp+2) = norm[k].face[2].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord_el+6));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord_el));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord_el+9));
			glEnd();

/* Triangle face 3 */

			*(norm_temp) = norm[k].face[3].x;
			*(norm_temp+1) = norm[k].face[3].y;
			*(norm_temp+2) = norm[k].face[3].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord_el+3));
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord_el+6));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord_el+9));
			glEnd();
		   }
   
/* Draw the wire frame around the mesh */
   
		   if( Outline_flag )
		   {
			glColor4fv( black );
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord_el+3));
				glVertex3dv((coord_el));
				glVertex3dv((coord_el+6));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord_el+3));
				glVertex3dv((coord_el+9));
				glVertex3dv((coord_el));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord_el+6));
				glVertex3dv((coord_el));
				glVertex3dv((coord_el+9));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord_el+3));
				glVertex3dv((coord_el+6));
				glVertex3dv((coord_el+9));
			glEnd();
		   }
		}

		if( input_color_flag )
		{
		     c0 = 8;
		     c1 = 8;
		     c2 = 8;
		     c3 = 8;
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
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord0_el+3));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord0_el));
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord0_el+6));
			glEnd();

/* Triangle face 1 */

			*(norm_temp) = norm0[k].face[1].x;
			*(norm_temp+1) = norm0[k].face[1].y;
			*(norm_temp+2) = norm0[k].face[1].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord0_el+3));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord0_el+9));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord0_el));
			glEnd();

/* Triangle face 2 */

			*(norm_temp) = norm0[k].face[2].x;
			*(norm_temp+1) = norm0[k].face[2].y;
			*(norm_temp+2) = norm0[k].face[2].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord0_el+6));
				glColor4fv(MeshColor[c0]);
				glVertex3dv((coord0_el));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord0_el+9));
			glEnd();

/* Triangle face 3 */

			*(norm_temp) = norm0[k].face[3].x;
			*(norm_temp+1) = norm0[k].face[3].y;
			*(norm_temp+2) = norm0[k].face[3].z;
			glBegin(GL_TRIANGLES);
				glNormal3fv(norm_temp);
				glColor4fv(MeshColor[c1]);
				glVertex3dv((coord0_el+3));
				glColor4fv(MeshColor[c2]);
				glVertex3dv((coord0_el+6));
				glColor4fv(MeshColor[c3]);
				glVertex3dv((coord0_el+9));
			glEnd();
		   }
   
/* Draw the wire frame around the mesh */
   
		   if( Outline_flag )
		   {
			glColor4fv( wire_color );
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord0_el+3));
				glVertex3dv((coord0_el));
				glVertex3dv((coord0_el+6));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord0_el+3));
				glVertex3dv((coord0_el+9));
				glVertex3dv((coord0_el));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord0_el+6));
				glVertex3dv((coord0_el));
				glVertex3dv((coord0_el+9));
			glEnd();
			glBegin(GL_LINE_LOOP);
				glVertex3dv((coord0_el+3));
				glVertex3dv((coord0_el+6));
				glVertex3dv((coord0_el+9));
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


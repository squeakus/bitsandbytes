/*
    This library function writes the resulting data for a finite element
    program which does analysis on a plate element.

		Updated 11/9/06

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plconst.h"
#include "plstruct.h"

extern int dof, nmat, nmode, numel, numnp, plane_stress_flag, plane_stress_read_flag;
extern int static_flag, element_stress_print_flag, gauss_stress_flag, flag_3D,
	flag_quad_element;

int plwriter ( BOUND bc, int *connect, double *coord, CURVATURE *curve,
	MDIM *curve_node, int *el_matl, double *force, int *id, MATL *matl,
	MOMENT *moment, MDIM *moment_node, char *name, STRAIN *strain,
	SDIM *strain_node, STRESS *stress, SDIM *stress_node, double *U, double *Uz_fib)
{
	int i,j,dum,check, node, name_length;
	char *ccheck;
	double fpointx, fpointy, fpointz;
	char out[30], stress_dat[40], stress_type[6];
	FILE *o3, *o4;

/* Every output file is named after the input file with
   a ".obr" extension */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(out,0,30*sizeof(char));

	ccheck = strncpy(out, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(out+name_length, ".opl", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o3 = fopen( out,"w" );

	fprintf( o3, "   numel numnp nmat nmode");
	if(plane_stress_read_flag) fprintf( o3, " plane_stress_flag");
	fprintf( o3, " (This is for the plate mesh file: %s)\n", name);
	fprintf( o3, "    %4d %4d %4d %4d", numel, numnp, nmat, nmode);
	if(plane_stress_read_flag) fprintf( o3, " %4d\n", plane_stress_flag);
	else fprintf( o3, "\n");
	fprintf( o3, " matl no., E mod., Poiss. Ratio, density, thick, shear fac.\n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o3, "  %4d    %12.6e %12.6e %12.6e %12.6e %12.6e \n ", i, matl[i].E,
		matl[i].nu, matl[i].rho, matl[i].thick, matl[i].shear);
	}

	fprintf( o3, "el no., connectivity, matl no. \n");

	if( !flag_quad_element )
	{
/* Triangle elements */
	   for( i = 0; i < numel; ++i )
	   {
		fprintf( o3, "%6d ",i);
		for( j = 0; j < npel3; ++j )
		{
			fprintf( o3, "%6d ",*(connect+npel3*i+j));
		}
		fprintf( o3, "   %3d\n",*(el_matl+i));
	   }
	}
	else
	{
	   for( i = 0; i < numel; ++i )
	   {
		fprintf( o3, "%6d ",i);
		for( j = 0; j < npel; ++j )
		{
			fprintf( o3, "%6d ",*(connect+npel*i+j));
		}
		fprintf( o3, "   %3d\n",*(el_matl+i));
	   }
	}

	fprintf( o3, "node no., coordinates \n");

	if( !flag_3D )
	{
	   for( i = 0; i < numnp; ++i )
	   {
		fpointx = *(coord+nsd*i) + *(U+ndof6*i) + *(U+ndof6*i+2)*(*(U+ndof6*i+4));
		fpointy = *(coord+nsd*i+1) + *(U+ndof6*i+1) - *(U+ndof6*i+2)*(*(U+ndof6*i+3));
		fprintf( o3, "%4d %14.6f %14.6f\n", i, fpointx, fpointy );
	   }
	}
	else
	{
	   for( i = 0; i < numnp; ++i )
	   {
#if 0
/* My original intention was to calculate the updated coordinates in the same way as the
   shell, but Uz_fib for the plate does not work the same way as for the shell.  Uz_fib,
   which represents the displacement in the fiber direction, is the same at each node for
   the shell, but not for the plate because a plate element's face normal determines
   the fiber direction, and this normal changes for each plate.
*/
		fpointx = *(coord+nsd*i) + *(U+ndof6*i) + *(Uz_fib+i)*(*(U+ndof6*i+4));
		fpointy = *(coord+nsd*i+1) + *(U+ndof6*i+1) - *(Uz_fib+i)*(*(U+ndof6*i+3));
		fpointz = *(coord+nsd*i+2) + *(U+ndof6*i+2);
#endif
/* Assuming the angles are global quantities, I will calculate the updated coordinates based
   on how I used to do it for the shell.
*/
		fpointx = *(coord+nsd*i) + *(U+ndof6*i) + *(U+ndof6*i+2)*(*(U+ndof6*i+4));
		fpointy = *(coord+nsd*i+1) + *(U+ndof6*i+1) - *(U+ndof6*i+2)*(*(U+ndof6*i+3));
		fpointz = *(coord+nsd*i+2) + *(U+ndof6*i+2);
		fprintf( o3, "%4d %14.6f %14.6f %14.6f\n", i, fpointx, fpointy, fpointz );
	   }
	}

	if( flag_3D )
	{
	   fprintf( o3, "prescribed displacement x: node  disp value \n");
	   for( i = 0; i < numnp; ++i )
	   {
		fprintf( o3, "%4d %14.6e\n", i, *(U+ndof6*i));
	   }
	   fprintf( o3, " -10 \n");

	   fprintf( o3, "prescribed displacement y: node  disp value \n");
	   for( i = 0; i < numnp; ++i )
	   {
		fprintf( o3, "%4d %14.6e\n", i, *(U+ndof6*i+1));
	   }
	   fprintf( o3, " -10 \n");
	}

	fprintf( o3, "prescribed displacement z: node  disp value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof6*i+2));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi x: node angle value\n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof6*i+3));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi y: node angle value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof6*i+4));
	}
	fprintf( o3, " -10 \n");

	if( flag_3D )
	{
	   fprintf( o3, "prescribed angle phi z: node angle value \n");
	   for( i = 0; i < numnp; ++i )
	   {
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof6*i+5));
	   }
	   fprintf( o3, " -10 \n");
	}

	fprintf( o3, "node with point load z and 2 moments phi x, phi y \n");

	if( static_flag)
	{
	    for( i = 0; i < bc.num_force[0] ; ++i )
	    {
		node = bc.force[i];
		if( flag_3D )
		{
			*(id+ndof6*node) = -1;
			*(id+ndof6*node+1) = -1;
			*(id+ndof6*node+5) = -1;
		}
		*(id+ndof6*node+2) = -1;
		*(id+ndof6*node+3) = -1;
		*(id+ndof6*node+4) = -1;
	    }
	}

	if( !flag_3D )
	{
	   for( i = 0; i < numnp; ++i )
	   {
		if( *(id+ndof6*i+2) < 0 || *(id+ndof6*i+3) < 0 || *(id+ndof6*i+4) < 0 )
		{
			fprintf( o3,"%4d",i);
			fprintf( o3," %16.4e  %16.4e  %16.4e ",
				*(force+ndof6*i+2), *(force+ndof6*i+3), *(force+ndof6*i+4));
			fprintf( o3, "\n");
		}
	   }
	}
	else
	{
	   for( i = 0; i < numnp; ++i )
	   {
		if( *(id+ndof6*i) < 0 || *(id+ndof6*i+1) < 0 || *(id+ndof6*i+2) < 0 ||
			*(id+ndof6*i+3) < 0 || *(id+ndof6*i+4) < 0 || *(id+ndof6*i+5) < 0 )
		{
			fprintf( o3,"%4d",i);
			fprintf( o3," %16.4e %16.4e %16.4e %16.4e %16.4e %16.4e",
				*(force+ndof6*i), *(force+ndof6*i+1), *(force+ndof6*i+2),
				*(force+ndof6*i+3), *(force+ndof6*i+4), *(force+ndof6*i+5));
			fprintf( o3, "\n");
		}
	   }
	}
	fprintf( o3, " -10\n");

/* For 2-D meshes */
	if( !flag_3D )
	{
	    fprintf( o3, "node no. with moment xx,yy,xy ");
	    fprintf( o3, "and stress vector in zx,yz\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",moment_node[i].xx);
		fprintf( o3,"%14.6e ",moment_node[i].yy);
		fprintf( o3,"%14.6e ",moment_node[i].xy);
		fprintf( o3,"%14.6e ",stress_node[i].zx);
		fprintf( o3,"%14.6e ",stress_node[i].yz);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with moment ");
	    fprintf( o3, "and principal moment I,II \n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",moment_node[i].I);
		fprintf( o3,"%14.6e ",moment_node[i].II);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with curvature xx,yy,xy ");
	    fprintf( o3, "and strain vector in zx,yz\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",curve_node[i].xx);
		fprintf( o3,"%14.6e ",curve_node[i].yy);
		fprintf( o3,"%14.6e ",curve_node[i].xy);
		fprintf( o3,"%14.6e ",strain_node[i].zx);
		fprintf( o3,"%14.6e ",strain_node[i].yz);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with curvature ");
	    fprintf( o3, "and principal curvature I,II \n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",curve_node[i].I);
		fprintf( o3,"%14.6e ",curve_node[i].II);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	}
/* For 3-D meshes */
	else
	{
	    fprintf( o3, "node no. with moment xx,yy,xy ");
	    fprintf( o3, "and stress vector in xx,yy,xy,zx,yz\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",moment_node[i].xx);
		fprintf( o3,"%14.6e ",moment_node[i].yy);
		fprintf( o3,"%14.6e ",moment_node[i].xy);
		fprintf( o3,"%14.6e ",stress_node[i].xx);
		fprintf( o3,"%14.6e ",stress_node[i].yy);
		fprintf( o3,"%14.6e ",stress_node[i].xy);
		fprintf( o3,"%14.6e ",stress_node[i].zx);
		fprintf( o3,"%14.6e ",stress_node[i].yz);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with moment ");
	    fprintf( o3, "and principal moment I,II and stress I,II,III\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",moment_node[i].I);
		fprintf( o3,"%14.6e ",moment_node[i].II);
		fprintf( o3,"%14.6e ",stress_node[i].I);
		fprintf( o3,"%14.6e ",stress_node[i].II);
		fprintf( o3,"%14.6e ",stress_node[i].III);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with curvature xx,yy,xy ");
	    fprintf( o3, "and strain vector in xx,yy,xy,zx,yz\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",curve_node[i].xx);
		fprintf( o3,"%14.6e ",curve_node[i].yy);
		fprintf( o3,"%14.6e ",curve_node[i].xy);
		fprintf( o3,"%14.6e ",strain_node[i].xx);
		fprintf( o3,"%14.6e ",strain_node[i].yy);
		fprintf( o3,"%14.6e ",strain_node[i].xy);
		fprintf( o3,"%14.6e ",strain_node[i].zx);
		fprintf( o3,"%14.6e ",strain_node[i].yz);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	    fprintf( o3, "node no. with curvature ");
	    fprintf( o3, "and principal curvature I,II and strain I,II,III\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",curve_node[i].I);
		fprintf( o3,"%14.6e ",curve_node[i].II);
		fprintf( o3,"%14.6e ",strain_node[i].I);
		fprintf( o3,"%14.6e ",strain_node[i].II);
		fprintf( o3,"%14.6e ",strain_node[i].III);
		fprintf( o3, "\n");
	    }
	    fprintf( o3, " -10 \n");
	}

	if(element_stress_print_flag)
	{

		memset(stress_type, 0, 6*sizeof(char));
		if(gauss_stress_flag)
		{
		    ccheck = strncpy(stress_type, "gauss", 5);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}
		else
		{
		    ccheck = strncpy(stress_type, "nodal", 5);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}

/* Open stress data output file */

		memset(stress_dat,0,40*sizeof(char));

		ccheck = strncpy(stress_dat, name, name_length);
		if(!ccheck) printf( " Problems with strncpy \n");

		ccheck = strncpy(stress_dat+name_length, ".str.opl", 8);
		if(!ccheck) printf( " Problems with strncpy \n");

		o4 = fopen( stress_dat,"w" );

/* For 2-D meshes */
		if( !flag_3D )
		{
		    if( !flag_quad_element )
		    {
/* Triangle elements */
			fprintf( o4, "element no. with moment ", stress_type);
			fprintf( o4, "xx,yy,xy and stress vector in zx,yz\n");
			for( i = 0; i < numel; ++i )
			{
				fprintf( o4,"%4d   ",i);
				fprintf( o4,"%14.6e ",moment[i].pt[0].xx);
				fprintf( o4,"%14.6e ",moment[i].pt[0].yy);
				fprintf( o4,"%14.6e ",moment[i].pt[0].xy);
				fprintf( o4,"%14.6e ",stress[i].pt[0].zx);
				fprintf( o4,"%14.6e ",stress[i].pt[0].yz);
				fprintf( o4, "\n");
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. with moment ", stress_type);
			fprintf( o4, "and principal moment I,II \n");
			for( i = 0; i < numel; ++i )
			{
				fprintf( o4,"%4d   ",i);
				fprintf( o4,"%14.6e ",moment[i].pt[0].I);
				fprintf( o4,"%14.6e ",moment[i].pt[0].II);
				fprintf( o4, "\n");
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. with curvature ", stress_type);
			fprintf( o4, "xx,yy,xy and strain vector in zx,yz \n");
			for( i = 0; i < numel; ++i )
			{
				fprintf( o4,"%4d   ",i);
				fprintf( o4,"%14.6e ",curve[i].pt[0].xx);
				fprintf( o4,"%14.6e ",curve[i].pt[0].yy);
				fprintf( o4,"%14.6e ",curve[i].pt[0].xy);
				fprintf( o4,"%14.6e ",strain[i].pt[0].zx);
				fprintf( o4,"%14.6e ",strain[i].pt[0].yz);
				fprintf( o4, "\n");
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. with curvature ", stress_type);
			fprintf( o4, "and principal curvature I,II \n");
			for( i = 0; i < numel; ++i )
			{
				fprintf( o4,"%4d   ",i);
				fprintf( o4,"%14.6e ",curve[i].pt[0].I);
				fprintf( o4,"%14.6e ",curve[i].pt[0].II);
				fprintf( o4, "\n");
			}
			fprintf( o4, " -10 \n");
		    }
		    else
		    {
/* Quadrilateral elements */
			fprintf( o4, "element no. and %5s pt. no. with moment ", stress_type);
			fprintf( o4, "xx,yy,xy and stress vector in zx,yz\n");
			for( i = 0; i < numel; ++i )
			{
			    for( j = 0; j < num_int; ++j )
			    {
				fprintf( o4,"%4d %4d  ",i,j);
				fprintf( o4,"%14.6e ",moment[i].pt[j].xx);
				fprintf( o4,"%14.6e ",moment[i].pt[j].yy);
				fprintf( o4,"%14.6e ",moment[i].pt[j].xy);
				fprintf( o4,"%14.6e ",stress[i].pt[j].zx);
				fprintf( o4,"%14.6e ",stress[i].pt[j].yz);
				fprintf( o4, "\n");
			    }
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. and %5s pt. no. with moment ", stress_type);
			fprintf( o4, "and principal moment I,II \n");
			for( i = 0; i < numel; ++i )
			{
			    for( j = 0; j < num_int; ++j )
			    {
				fprintf( o4,"%4d %4d  ",i,j);
				fprintf( o4,"%14.6e ",moment[i].pt[j].I);
				fprintf( o4,"%14.6e ",moment[i].pt[j].II);
				fprintf( o4, "\n");
			    }
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. and %5s pt. no. with curvature ", stress_type);
			fprintf( o4, "xx,yy,xy and strain vector in zx,yz \n");
			for( i = 0; i < numel; ++i )
			{
			    for( j = 0; j < num_int; ++j )
			    {
				fprintf( o4,"%4d %4d  ",i,j);
				fprintf( o4,"%14.6e ",curve[i].pt[j].xx);
				fprintf( o4,"%14.6e ",curve[i].pt[j].yy);
				fprintf( o4,"%14.6e ",curve[i].pt[j].xy);
				fprintf( o4,"%14.6e ",strain[i].pt[j].zx);
				fprintf( o4,"%14.6e ",strain[i].pt[j].yz);
				fprintf( o4, "\n");
			    }
			}
			fprintf( o4, " -10 \n");
			fprintf( o4, "element no. and %5s pt. no. with curvature ", stress_type);
			fprintf( o4, "and principal curvature I,II \n");
			for( i = 0; i < numel; ++i )
			{
			    for( j = 0; j < num_int; ++j )
			    {
				fprintf( o4,"%4d %4d  ",i,j);
				fprintf( o4,"%14.6e ",curve[i].pt[j].I);
				fprintf( o4,"%14.6e ",curve[i].pt[j].II);
				fprintf( o4, "\n");
			    }
			}
			fprintf( o4, " -10 \n");
		    }
		}
/* For 3-D meshes */
		else
		{
		    if( !flag_quad_element )
		    {
/* Triangle elements */
                        fprintf( o4, "element no. with moment ", stress_type);
                        fprintf( o4, "xx,yy,xy and stress vector in xx,yy,xy,zx,yz\n");
                        for( i = 0; i < numel; ++i )
                        {
                                fprintf( o4,"%4d   ",i);
                                fprintf( o4,"%14.6e ",moment[i].pt[0].xx);
                                fprintf( o4,"%14.6e ",moment[i].pt[0].yy);
                                fprintf( o4,"%14.6e ",moment[i].pt[0].xy);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].xx);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].yy);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].xy);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].zx);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].yz);
                                fprintf( o4, "\n");
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. with moment ", stress_type);
                        fprintf( o4, "and principal moment I,II and stress I,II,III\n");
                        for( i = 0; i < numel; ++i )
                        {
                                fprintf( o4,"%4d   ",i);
                                fprintf( o4,"%14.6e ",moment[i].pt[0].I);
                                fprintf( o4,"%14.6e ",moment[i].pt[0].II);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].I);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].II);
                                fprintf( o4,"%14.6e ",stress[i].pt[0].III);
                                fprintf( o4, "\n");
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. with curvature ", stress_type);
                        fprintf( o4, "xx,yy,xy and strain vector in xx,yy,xy,zx,yz \n");
                        for( i = 0; i < numel; ++i )
                        {
                                fprintf( o4,"%4d   ",i);
                                fprintf( o4,"%14.6e ",curve[i].pt[0].xx);
                                fprintf( o4,"%14.6e ",curve[i].pt[0].yy);
                                fprintf( o4,"%14.6e ",curve[i].pt[0].xy);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].xx);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].yy);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].xy);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].zx);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].yz);
                                fprintf( o4, "\n");
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. with curvature ", stress_type);
                        fprintf( o4, "and principal curvature I,II and strain I,II,III\n");
                        for( i = 0; i < numel; ++i )
                        {
                                fprintf( o4,"%4d   ",i);
                                fprintf( o4,"%14.6e ",curve[i].pt[0].I);
                                fprintf( o4,"%14.6e ",curve[i].pt[0].II);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].I);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].II);
                                fprintf( o4,"%14.6e ",strain[i].pt[0].III);
                                fprintf( o4, "\n");
                        }
                        fprintf( o4, " -10 \n");
		    }
		    else
		    {
/* Quadrilateral elements */
                        fprintf( o4, "element no. and %5s pt. no. with moment ", stress_type);
                        fprintf( o4, "xx,yy,xy and stress vector in xx,yy,xy,zx,yz\n");
                        for( i = 0; i < numel; ++i )
                        {
                            for( j = 0; j < num_int; ++j )
                            {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",moment[i].pt[j].xx);
                                fprintf( o4,"%14.6e ",moment[i].pt[j].yy);
                                fprintf( o4,"%14.6e ",moment[i].pt[j].xy);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].xx);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].yy);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].xy);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].zx);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].yz);
                                fprintf( o4, "\n");
                            }
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. and %5s pt. no. with moment ", stress_type);
                        fprintf( o4, "and principal moment I,II and stress I,II,III\n");
                        for( i = 0; i < numel; ++i )
                        {
                            for( j = 0; j < num_int; ++j )
                            {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",moment[i].pt[j].I);
                                fprintf( o4,"%14.6e ",moment[i].pt[j].II);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].I);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].II);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].III);
                                fprintf( o4, "\n");
                            }
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. and %5s pt. no. with curvature ", stress_type);
                        fprintf( o4, "xx,yy,xy and strain vector in xx,yy,xy,zx,yz \n");
                        for( i = 0; i < numel; ++i )
                        {
                            for( j = 0; j < num_int; ++j )
                            {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",curve[i].pt[j].xx);
                                fprintf( o4,"%14.6e ",curve[i].pt[j].yy);
                                fprintf( o4,"%14.6e ",curve[i].pt[j].xy);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].xx);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].yy);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].xy);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].zx);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].yz);
                                fprintf( o4, "\n");
                            }
                        }
                        fprintf( o4, " -10 \n");
                        fprintf( o4, "element no. and %5s pt. no. with curvature ", stress_type);
                        fprintf( o4, "and principal curvature I,II and strain I,II,III\n");
                        for( i = 0; i < numel; ++i )
                        {
                            for( j = 0; j < num_int; ++j )
                            {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",curve[i].pt[j].I);
                                fprintf( o4,"%14.6e ",curve[i].pt[j].II);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].I);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].II);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].III);
                                fprintf( o4, "\n");
                            }
                        }
                        fprintf( o4, " -10 \n");
		    }
		}
	}

	return 1;
}


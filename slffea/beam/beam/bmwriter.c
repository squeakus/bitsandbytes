/*
    This library function writes the resulting data for a finite element
    program which does analysis on a beam 

		Updated 1/12/05

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmconst.h"
#include "bmstruct.h"

extern int dof, nmat, nmode, numel, numnp;
extern int static_flag, gauss_stress_flag;

int bmwriter ( double *axis_z, BOUND bc, int *connect, double *coord, CURVATURE *curve,
	int *el_matl, int *el_type, double *force, int *id, MATL *matl, MOMENT *moment,
	char *name, STRAIN *strain, STRESS *stress, double *U)
{
	int i,j,dum,check, node, node0, node1, name_length;
	char *ccheck;
	double fpointx, fpointy, fpointz;
	char out[30], stress_type[6];
	FILE *o3, *o4;

/* Every output file is named after the input file with
   a ".obm" extension */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(out,0,30*sizeof(char));

	ccheck = strncpy(out, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(out+name_length, ".obm", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o3 = fopen( out,"w" );

	fprintf( o3, "   numel numnp nmat nmode (This is for the beam mesh file: %s)\n", name);
	fprintf( o3, "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nmode);
	fprintf( o3, "matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz, AreaSy, AreaSz\n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o3, "  %4d   %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e",
		i,matl[i].E, matl[i].nu, matl[i].rho, matl[i].area, matl[i].Iy, matl[i].Iz);
	   if( matl[i].extraArea > 0 )
	   {
		fprintf( o3, " %12.6e %12.6e\n", matl[i].areaSy, matl[i].areaSz);
	   }
	   else
	   {
		fprintf( o3, "\n");
	   }
	}

	fprintf( o3, "el no., connectivity, matl no., element type\n");
	for( i = 0; i < numel; ++i )
	{
	   fprintf( o3, "%6d ",i);
	   for( j = 0; j < npel; ++j )
	   {
		fprintf( o3, "%6d ",*(connect+npel*i+j));
	   }
	   fprintf( o3, "   %3d",*(el_matl+i));
	   fprintf( o3, "   %3d\n",*(el_type+i));
	}

	fprintf( o3, "node no., coordinates \n");
	for( i = 0; i < numnp; ++i )
	{
	   fpointx = *(coord+nsd*i) + *(U+ndof*i);
	   fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1);
	   fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
	   fprintf( o3, "%4d %14.6f %14.6f %14.6f\n", i,
		fpointx, fpointy, fpointz);
	}

	fprintf( o3, "element with specified local z axis: x, y, z component \n");
	for( i = 0; i < numel; ++i )
	{
		fprintf( o3, "%4d %12.6f %12.6f %12.6f\n", i, *(axis_z+nsd*i),
			*(axis_z+nsd*i+1), *(axis_z+nsd*i+2));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed displacement x: node  disp value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed displacement y: node  disp value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+1));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed displacement z: node  disp value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+2));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi x: node angle value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+3));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi y: node angle value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+4));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi z: node angle value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+5));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "node with point load x, y, z and 3 moments phi x, phi y, phi z \n");

	if( static_flag)
	{
	    for( i = 0; i < bc.num_force[0] ; ++i )
	    {
		node = bc.force[i];
		*(id+ndof*node) = -1;
		*(id+ndof*node+1) = -1; 
		*(id+ndof*node+2) = -1; 
		*(id+ndof*node+3) = -1; 
		*(id+ndof*node+4) = -1; 
		*(id+ndof*node+5) = -1;
	    }

	    for( i = 0; i < bc.num_dist_load[0] ; ++i )
	    {
		node0 = *(connect + npel*bc.dist_load[i]);
		node1 = *(connect + npel*bc.dist_load[i]+1);

		*(id+ndof*node0) = -1;
		*(id+ndof*node0+1) = -1;
		*(id+ndof*node0+2) = -1;
		*(id+ndof*node0+3) = -1;
		*(id+ndof*node0+4) = -1;
		*(id+ndof*node0+5) = -1;

		*(id+ndof*node1) = -1;
		*(id+ndof*node1+1) = -1;
		*(id+ndof*node1+2) = -1;
		*(id+ndof*node1+3) = -1;
		*(id+ndof*node1+4) = -1;
		*(id+ndof*node1+5) = -1;
	    }
	}

	for( i = 0; i < numnp; ++i )
	{
	   if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0 ||
		*(id+ndof*i+3) < 0 || *(id+ndof*i+4) < 0 || *(id+ndof*i+5) < 0 )
	   {
		fprintf( o3,"%4d",i);
		for( j = 0; j < ndof; ++j )
		{
			fprintf( o3," %14.6e ",*(force+ndof*i+j));
		}
		fprintf( o3, "\n");
	   }
	}
	fprintf( o3, " -10\n");

	fprintf( o3, "element with distributed load in local beam y and z coordinates \n");
	fprintf( o3, " -10\n");

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

	fprintf( o3, "element no. and %5s pt. no. with ", stress_type);
	fprintf( o3, "local stress xx,xy,zx and moment xx,yy,zz\n");
	for( i = 0; i < numel; ++i )
	{
	   for( j = 0; j < num_int; ++j )
	   {
		fprintf( o3,"%4d %4d  ",i,j);
		fprintf( o3,"%14.6e ",stress[i].pt[j].xx);
		fprintf( o3,"%14.6e ",stress[i].pt[j].xy);
		fprintf( o3,"%14.6e ",stress[i].pt[j].zx);
		fprintf( o3,"%14.6e ",moment[i].pt[j].xx);
		fprintf( o3,"%14.6e ",moment[i].pt[j].yy);
		fprintf( o3,"%14.6e ",moment[i].pt[j].zz);
		fprintf( o3, "\n");
	    }
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "element no. and %5s pt. no. with ", stress_type);
	fprintf( o3, "local strain xx,xy,zx and curvature xx,yy,zz\n");
	for( i = 0; i < numel; ++i )
	{
	   for( j = 0; j < num_int; ++j )
	   {
		fprintf( o3,"%4d %4d  ",i,j);
		fprintf( o3,"%14.6e ",strain[i].pt[j].xx);
		fprintf( o3,"%14.6e ",strain[i].pt[j].xy);
		fprintf( o3,"%14.6e ",strain[i].pt[j].zx);
		fprintf( o3,"%14.6e ",curve[i].pt[j].xx);
		fprintf( o3,"%14.6e ",curve[i].pt[j].yy);
		fprintf( o3,"%14.6e ",curve[i].pt[j].zz);
		fprintf( o3, "\n");
	   }
	}
	fprintf( o3, " -10 \n");

	return 1;
}


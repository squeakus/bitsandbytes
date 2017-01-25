/*
    This library function writes the resulting data for a finite element
    program which does analysis on a brick.

		Updated 5/2/05

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "brconst.h"
#include "brstruct.h"

extern int dof, nmat, nmode, numel, numnp;
extern int static_flag, element_stress_print_flag, gauss_stress_flag;

int brwriter ( BOUND bc, int *connect, double *coord, int *el_matl, double *force,
	int *id, MATL *matl, char *name, STRAIN *strain, SDIM *strain_node,
	STRESS *stress, SDIM *stress_node, double *U)
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

	ccheck = strncpy(out+name_length, ".obr", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o3 = fopen( out,"w" );

	fprintf( o3, "   numel numnp nmat nmode (This is for the brick mesh file: %s)\n", name);
	fprintf( o3, "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nmode);
	fprintf( o3, "matl no., E modulus, Poisson Ratio, density \n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o3, "  %4d    %12.6e %12.6e %12.6e\n",i,matl[i].E, matl[i].nu, matl[i].rho);
	}

	fprintf( o3, "el no., connectivity, matl no. \n");
	for( i = 0; i < numel; ++i )
	{
	   fprintf( o3, "%6d ",i);
	   for( j = 0; j < npel; ++j )
	   {
		fprintf( o3, "%6d ",*(connect+npel*i+j));
	   }
	   fprintf( o3, "   %3d\n",*(el_matl+i));
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

	fprintf( o3, "node with point load and load vector in x,y,z \n");

	if( static_flag)
	{
	    for( i = 0; i < bc.num_force[0] ; ++i )
	    {
		node = bc.force[i];
		*(id+ndof*node) = -1;
		*(id+ndof*node+1) = -1;
		*(id+ndof*node+2) = -1;
	    }
	}

	for( i = 0; i < numnp; ++i )
	{
	   if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0 )
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

	fprintf( o3, "node no. with stress ");
	fprintf( o3, "and stress vector in xx,yy,zz,xy,zx,yz \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",stress_node[i].xx);
		fprintf( o3,"%14.6e ",stress_node[i].yy);
		fprintf( o3,"%14.6e ",stress_node[i].zz);
		fprintf( o3,"%14.6e ",stress_node[i].xy);
		fprintf( o3,"%14.6e ",stress_node[i].zx);
		fprintf( o3,"%14.6e ",stress_node[i].yz);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with stress ");
	fprintf( o3, "and principal stress I,II,III \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",stress_node[i].I);
		fprintf( o3,"%14.6e ",stress_node[i].II);
		fprintf( o3,"%14.6e ",stress_node[i].III);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with strain ");
	fprintf( o3, "and strain vector in xx,yy,zz,xy,zx,yz \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",strain_node[i].xx);
		fprintf( o3,"%14.6e ",strain_node[i].yy);
		fprintf( o3,"%14.6e ",strain_node[i].zz);
		fprintf( o3,"%14.6e ",strain_node[i].xy);
		fprintf( o3,"%14.6e ",strain_node[i].zx);
		fprintf( o3,"%14.6e ",strain_node[i].yz);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with strain ");
	fprintf( o3, "and principal strain I,II,III \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",strain_node[i].I);
		fprintf( o3,"%14.6e ",strain_node[i].II);
		fprintf( o3,"%14.6e ",strain_node[i].III);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");

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

		ccheck = strncpy(stress_dat+name_length, ".str.obr", 8);
		if(!ccheck) printf( " Problems with strncpy \n");

		o4 = fopen( stress_dat,"w" );

		fprintf( o4, "element no. and %5s pt. no. with stress ", stress_type);
		fprintf( o4, "and stress vector in xx,yy,zz,xy,zx,yz \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",stress[i].pt[j].xx);
			fprintf( o4,"%14.6e ",stress[i].pt[j].yy);
			fprintf( o4,"%14.6e ",stress[i].pt[j].zz);
			fprintf( o4,"%14.6e ",stress[i].pt[j].xy);
			fprintf( o4,"%14.6e ",stress[i].pt[j].zx);
			fprintf( o4,"%14.6e ",stress[i].pt[j].yz);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with stress ", stress_type);
		fprintf( o4, "and principal stress I,II,III \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",stress[i].pt[j].I);
			fprintf( o4,"%14.6e ",stress[i].pt[j].II);
			fprintf( o4,"%14.6e ",stress[i].pt[j].III);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with strain ", stress_type);
		fprintf( o4, "and strain vector in xx,yy,zz,xy,zx,yz \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",strain[i].pt[j].xx);
			fprintf( o4,"%14.6e ",strain[i].pt[j].yy);
			fprintf( o4,"%14.6e ",strain[i].pt[j].zz);
			fprintf( o4,"%14.6e ",strain[i].pt[j].xy);
			fprintf( o4,"%14.6e ",strain[i].pt[j].zx);
			fprintf( o4,"%14.6e ",strain[i].pt[j].yz);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with strain ", stress_type);
		fprintf( o4, "and principal strain I,II,III \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",strain[i].pt[j].I);
			fprintf( o4,"%14.6e ",strain[i].pt[j].II);
			fprintf( o4,"%14.6e ",strain[i].pt[j].III);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
	}

	return 1;
}


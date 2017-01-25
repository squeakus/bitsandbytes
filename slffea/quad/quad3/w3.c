/*
    This library function writes the resulting data for a finite element
    program which does analysis on a quadrilateral element 

		Updated 12/7/00

    SLFFEA source file
    Version:  1.1
    Copyright (C) 1999  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qd3const.h"
#include "qd3struct.h"

extern int dof, nmat, nmode, numed, numel, numnp, plane_stress_flag;
extern int static_flag, element_stress_print_flag, gauss_stress_flag;

int qd3writer ( BOUND bc, int *connect, int *edge_connect, int *el_edge_connect,
	double *coord, int *el_matl, double *force, int *id, MATL *matl, char *name,
	STRAIN *strain, SDIM *strain_node, STRESS *stress, SDIM *stress_node,
	double *EMedge, double *EMnode)
{
        int i,j,dum,check, node, name_length;
	char *ccheck;
	double fpointx, fpointy;
	char out[30], stress_dat[40], stress_type[6];
	FILE *o3, *o4;

/* Every output file is named after the input file with
   a ".oqd" extension */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(out,0,30*sizeof(char));

	ccheck = strncpy(out, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(out+name_length, ".oqd", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

        o3 = fopen( out,"w" );

        fprintf( o3, "   numed numel numnp nmat nmode plane_stress_flag");
        fprintf( o3, " (This is for the quad3 mesh file: %s)\n", name);
        fprintf( o3, "    %4d %4d %4d %4d %4d %4d\n ",
		numed, numel,numnp,nmat,nmode,plane_stress_flag);
        fprintf( o3, "matl no., permeability, permittivity \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o3, "  %4d    %9.4f %9.4f\n ",i,
		matl[i].eta, matl[i].nu);
        }

        fprintf( o3, "edge no., node connectivity, matl no.\n");
	for( i = 0; i < numed; ++i )
	{
           fprintf( o3, "%6d ",i);
	   for( j = 0; j < nped; ++j )
	   {
		fprintf( o3, "%4d ",*(edge_connect+nped*i+j));
	   }
	   fprintf( o3,"\n");
	}
	printf( "\n");

        fprintf( o3, "el no., node connectivity, edge connectivity, matl no.\n");
        for( i = 0; i < numel; ++i )
        {
           fprintf( o3, "%6d ",i);
           for( j = 0; j < npel; ++j )
           {
                fprintf( o3, "%6d ",*(connect+npel*i+j));
           }
           fprintf( o3, "    ");
           for( j = 0; j < epel; ++j )
           {
                fprintf( o3, "%6d ",*(el_edge_connect+epel*i+j));
           }
           fprintf( o3, "   %3d\n",*(el_matl+i));
        }

        fprintf( o3, "node no., coordinates \n");
        for( i = 0; i < numnp; ++i )
        {
           fpointx = *(coord+nsd*i);
           fpointy = *(coord+nsd*i+1);
           fprintf( o3, "%4d %14.6f %14.6f\n", i, fpointx, fpointy );
        }

        fprintf( o3, "prescribed edge: edge value\n");
        for( i = 0; i < numed; ++i )
        {
                fprintf( o3, "%4d %14.6e\n", i,
			*(EMedge+edof*i));
        }
        fprintf( o3, " -10 \n");

        fprintf( o3, "node with point load and load vector in x,y \n");

	if( static_flag)
	{
            for( i = 0; i < bc.num_force[0] ; ++i )
            {
		node = bc.force[i];
		*(id+edof*node) = -1;
		*(id+edof*node+1) = -1;
            }
        }

        for( i = 0; i < numnp; ++i )
        {
           if( *(id+edof*i) < 0 || *(id+edof*i+1) < 0 )
           {
           	fprintf( o3,"%4d",i);
           	for( j = 0; j < ndof; ++j )
           	{
               		fprintf( o3," %16.4e ",*(force+ndof*i+j));
           	}
           	fprintf( o3, "\n");
           }
        }
        fprintf( o3, " -10\n");

        fprintf( o3, "node no. with stress ");
        fprintf( o3, "and stress vector in xx,yy,xy \n");
	for( i = 0; i < numnp; ++i )
	{
                fprintf( o3,"%4d  ",i);
           	fprintf( o3,"%14.6e ",stress_node[i].xx);
           	fprintf( o3,"%14.6e ",stress_node[i].yy);
           	fprintf( o3,"%14.6e ",stress_node[i].xy);
           	fprintf( o3, "\n");
	}
        fprintf( o3, " -10 \n");
        fprintf( o3, "node no. with stress ");
        fprintf( o3, "and principal stress vector I,II \n");
	for( i = 0; i < numnp; ++i )
	{
                fprintf( o3,"%4d  ",i);
           	fprintf( o3,"%14.6e ",stress_node[i].I);
           	fprintf( o3,"%14.6e ",stress_node[i].II);
           	fprintf( o3, "\n");
	}
        fprintf( o3, " -10 \n");
        fprintf( o3, "node no. with strain ");
        fprintf( o3, "and strain vector in xx,yy,xy \n");
	for( i = 0; i < numnp; ++i )
	{
                fprintf( o3,"%4d  ",i);
           	fprintf( o3,"%14.6e ",strain_node[i].xx);
           	fprintf( o3,"%14.6e ",strain_node[i].yy);
           	fprintf( o3,"%14.6e ",strain_node[i].xy);
           	fprintf( o3, "\n");
	}
        fprintf( o3, " -10 \n");
        fprintf( o3, "node no. with strain ");
        fprintf( o3, "and principal strain vector I,II \n");
	for( i = 0; i < numnp; ++i )
	{
                fprintf( o3,"%4d  ",i);
           	fprintf( o3,"%14.6e ",strain_node[i].I);
           	fprintf( o3,"%14.6e ",strain_node[i].II);
           	fprintf( o3, "\n");
	}
        fprintf( o3, " -10 \n");

	if(element_stress_print_flag)
	{

		if(gauss_stress_flag)
		{
		    ccheck = strncpy(stress_type, "gauss ", 6);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}
		else
		{
		    ccheck = strncpy(stress_type, "nodal ", 6);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}

/* Open stress data output file */

		memset(stress_dat,0,40*sizeof(char));

		ccheck = strncpy(stress_dat, name, name_length);
		if(!ccheck) printf( " Problems with strncpy \n");

		ccheck = strncpy(stress_dat+name_length, ".str.oqd", 8);
		if(!ccheck) printf( " Problems with strncpy \n");

        	o4 = fopen( stress_dat,"w" );

                fprintf( o4, "element no. and %6spt. no. with stress ", stress_type);
                fprintf( o4, "and stress vector in xx,yy,xy \n");
                for( i = 0; i < numel; ++i )
                {
                   for( j = 0; j < num_int; ++j )
                   {
                   	fprintf( o4,"%4d %4d  ",i,j);
                   	fprintf( o4,"%14.6e ",stress[i].pt[j].xx);
                   	fprintf( o4,"%14.6e ",stress[i].pt[j].yy);
                   	fprintf( o4,"%14.6e ",stress[i].pt[j].xy);
                   	fprintf( o4, "\n");
	           }
                }
                fprintf( o4, " -10 \n");
                fprintf( o4, "element no. and %6spt. no. with stress ", stress_type);
                fprintf( o4, "and principal stress vector I,II \n");
                for( i = 0; i < numel; ++i )
                {
                   for( j = 0; j < num_int; ++j )
                   {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].I);
                                fprintf( o4,"%14.6e ",stress[i].pt[j].II);
                                fprintf( o4, "\n");
                   }
                }
                fprintf( o4, " -10 \n");
                fprintf( o4, "element no. and %6spt. no. with strain ", stress_type);
                fprintf( o4, "and strain vector in xx,yy,xy \n");
                for( i = 0; i < numel; ++i )
                {
                   for( j = 0; j < num_int; ++j )
                   {
                   	fprintf( o4,"%4d %4d  ",i,j);
                   	fprintf( o4,"%14.6e ",strain[i].pt[j].xx);
                   	fprintf( o4,"%14.6e ",strain[i].pt[j].yy);
                   	fprintf( o4,"%14.6e ",strain[i].pt[j].xy);
                   	fprintf( o4, "\n");
                   }
                }
                fprintf( o4, " -10 \n");
                fprintf( o4, "element no. and %6spt. no. with strain ", stress_type);
                fprintf( o4, "and principal strain vector I,II \n");
                for( i = 0; i < numel; ++i )
                {
                   for( j = 0; j < num_int; ++j )
                   {
                                fprintf( o4,"%4d %4d  ",i,j);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].I);
                                fprintf( o4,"%14.6e ",strain[i].pt[j].II);
                                fprintf( o4, "\n");
                   }
                }
                fprintf( o4, " -10 \n");
	}

        return 1;
}


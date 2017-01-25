/*
    This library function writes the resulting data for a finite element
    program which does analysis on a truss 

		Updated 7/10/02

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
#include "tsconst.h"
#include "tsstruct.h"

extern int dof, nmat, nmode, numel, numnp;
extern int standard_flag;

int tswriter ( BOUND bc, int *connect, double *coord, int *el_matl, double *force,
	int *id, MATL *matl, char *name, STRAIN *strain, STRESS *stress, double *U)
{
        int i,j,dum,check, node, name_length;
	char *ccheck;
	double fpointx, fpointy, fpointz;
	char out[30];
	FILE *o3;

/* Every output file is named after the input file with
   a ".ots" extension */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(out,0,30*sizeof(char));

	ccheck = strncpy(out, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(out+name_length, ".ots", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

        o3 = fopen( out,"w" );

        fprintf( o3, "   numel numnp nmat nmode (This is for the truss mesh file: %s)\n", name);
        fprintf( o3, "    %4d %4d %4d %4d\n ", numel,numnp,nmat,nmode);
        fprintf( o3, "matl no., E modulus, density, Area \n");
        for( i = 0; i < nmat; ++i )
        {
           fprintf( o3, "  %4d    %12.6e %12.6e %12.6e\n",i,matl[i].E, matl[i].rho, matl[i].area);
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

	if( standard_flag)
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

/* Open stress data output file */

	fprintf( o3, "element no. with stress and tensile stress vector \n");
	for( i = 0; i < numel; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",stress[i].xx);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "element no. with strain and tensile strain vector \n");
	for( i = 0; i < numel; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",strain[i].xx);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");

        return 1;
}


/*
    This library function reads in data for a finite element
    program which does analysis on a quadrilateral element

		Updated 8/6/04

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

extern int dof, EMdof, nmat, nmode, numed, numel, numnp, plane_stress_flag;
extern int stress_read_flag, element_stress_read_flag;

int qd3reader( BOUND bc, int *connect, int *edge_connect, int *el_edge_connect,
	double *coord, int *el_matl, double *force, MATL *matl, char *name,
	FILE *o1, STRESS *stress, SDIM *stress_node, double *EMedge)
{
	int i,j,dum,dum2, name_length;
	char *ccheck;
	char buf[ BUFSIZ ];
	char text, stress_dat[30];
	FILE *o4;

	if(element_stress_read_flag)
	{
/* Open stress data output file */

		name_length = strlen(name);
		if( name_length > 25) name_length = 25;

		memset(stress_dat,0,30*sizeof(char));

		ccheck = strncpy(stress_dat, name, name_length);
		if(!ccheck) printf( " Problems with strncpy \n");

		ccheck = strncpy(stress_dat+name_length, ".str", 4);
		if(!ccheck) printf( " Problems with strncpy \n");

		o4 = fopen( stress_dat,"r" );
		if(o4 == NULL ) {
			printf("Can't find file %30s\n",stress_dat);
			element_stress_read_flag = 0;
		}
	}

	printf( "number of elements:%d nodes:%d materials:%d modes:%d dof:%d EMdof:%d\n",
		numel,numnp,nmat,nmode,dof,EMdof);
	printf( "Plane Theory :%d \n",plane_stress_flag);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < nmat; ++i )
	{
	   fscanf( o1, "%d ",&dum);
	   printf( "material (%3d) permittivity, permeability, oper. fq.",dum);
	   fscanf( o1, " %lf %lf %lf\n", &matl[dum].eta, &matl[dum].nu, &matl[dum].ko);
	   printf( " %7.3e %7.3e %7.3e\n", matl[dum].eta, matl[dum].nu, matl[dum].ko);
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numed; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "connectivity for edge (%4d) ",dum);
	   for( j = 0; j < nped; ++j )
	   {
		fscanf( o1, "%d",(edge_connect+nped*dum+j));
		printf( "%4d ",*(edge_connect+nped*dum+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numel; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "connectivity for element (%4d) ",dum);
	   for( j = 0; j < npel; ++j )
	   {
		fscanf( o1, "%d",(connect+npel*dum+j));
		printf( "%4d ",*(connect+npel*dum+j));
	   }
	   printf( "    ");
	   for( j = 0; j < epel; ++j )
	   {
		fscanf( o1, "%d",(el_edge_connect+epel*dum+j));
		printf( "%4d ",*(el_edge_connect+epel*dum+j));
	   }
	   fscanf( o1,"%d\n",(el_matl+dum));
	   printf( " with matl %3d\n",*(el_matl+dum));
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numnp; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "coordinate (%d) ",dum);
	   printf( "coordinates ");
	   for( j = 0; j < nsd; ++j )
	   {
		fscanf( o1, "%lf ",(coord+nsd*dum+j));
		printf( "%9.6e ",*(coord+nsd*dum+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	dum= 0;
	fscanf( o1,"%d",&bc.edge[dum]);
	printf( "edge (%4d) has a prescribed value of: ",bc.edge[dum]);
	while( bc.edge[dum] > -1 )
	{
		fscanf( o1,"%lf\n%d",(EMedge+ndof*bc.edge[dum]),
			&bc.edge[dum+1]);
		printf( "%14.6e\n",*(EMedge+ndof*bc.edge[dum]));
		printf( "edge (%4d) has a prescribed value of: ",
			bc.edge[dum+1]);
		++dum;
	}
	bc.num_edge[0]=dum;
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	printf("force vector for node: ");
	fscanf( o1,"%d",&bc.force[dum]);
	printf( "(%4d)",bc.force[dum]);
	while( bc.force[dum] > -1 )
	{
	   for( j = 0; j < ndof; ++j )
	   {
		fscanf( o1,"%lf ",(force+ndof*bc.force[dum]+j));
		printf("%14.6e ",*(force+ndof*bc.force[dum]+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	   printf("force vector for node: ");
	   ++dum;
	   fscanf( o1,"%d",&bc.force[dum]);
	   printf( "(%4d)",bc.force[dum]);
	}
	bc.num_force[0]=dum;
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	if(stress_read_flag)
	{
	   printf("stress for node: ");
	   fscanf( o1,"%d",&dum);
	   printf( "(%4d)",dum);
	   while( dum > -1 )
	   {
		fscanf( o1,"%lf ",&stress_node[dum].xx);
		fscanf( o1,"%lf ",&stress_node[dum].yy);
		fscanf( o1,"%lf ",&stress_node[dum].xy);
		printf(" %12.5e",stress_node[dum].xx);
		printf(" %12.5e",stress_node[dum].yy);
		printf(" %12.5e",stress_node[dum].xy);
		fscanf( o1,"\n");
		printf( "\n");
		printf("stress for node: ");
		fscanf( o1,"%d",&dum);
		printf( "(%4d)",dum);
	   }
	}
	printf( "\n\n");

	if(element_stress_read_flag)
	{
	   fgets( buf, BUFSIZ, o4 );
	   printf( "\n\n");
	   printf("stress for ele: ");
	   fscanf( o4,"%d",&dum);
	   printf( "(%4d)",dum);
	   while( dum > -1 )
	   {
		fscanf( o4,"%d",&dum2);
		printf( " node (%1d)",dum2);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].xx);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].yy);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].xy);
		printf(" %12.5e",stress[dum].pt[dum2].xx);
		printf(" %12.5e",stress[dum].pt[dum2].yy);
		printf(" %12.5e",stress[dum].pt[dum2].xy);
		fscanf( o4,"\n");
		printf( "\n");
		printf("stress for ele: ");
		fscanf( o4,"%d",&dum);
		printf( "(%4d)",dum);
	   }
	}
	printf( "\n\n");

	return 1;
}


/*
    This library function reads in data for a finite element
    program which does analysis on a thermal brick element.

		Updated 9/30/06

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
#include "../brick/brconst.h"
#include "br2struct.h"

extern int Tdof, nmat, numel, numel_film, numnp;
extern int stress_read_flag, element_stress_read_flag;

int br2reader( BOUND bc, int *connect, int *connect_film, double *coord, int *el_matl,
	int *el_matl_film, double *force, double *heat_el, double *heat_node, MATL *matl,
	char *name, FILE *o1, double *Q, STRESS *stress, SDIM *stress_node, double *T,
	double *TB, double *U)
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

	printf( "number of elements:%d nodes:%d materials:%d film surfaces: %d Tdof:%d\n",
		numel,numnp,nmat,numel_film, Tdof);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < nmat; ++i )
	{
	   fscanf( o1, "%d ",&dum);
	   printf( "material (%3d) matl no., ther cond x, y, z, ther expan x, y, z,",dum);
	   printf( " film coeff, Emod x, y, z and Poi.Rat. xy, xz, yz\n",dum);
	   fscanf( o1, " %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
		&matl[dum].thrml_cond.x, &matl[dum].thrml_cond.y, &matl[dum].thrml_cond.z,
		&matl[dum].thrml_expn.x, &matl[dum].thrml_expn.y, &matl[dum].thrml_expn.z,
		&matl[dum].film,
		&matl[dum].E.x, &matl[dum].E.y, &matl[dum].E.z,
		&matl[dum].nu.xy, &matl[dum].nu.xz, &matl[dum].nu.yz);
	   printf( " %7.3e %7.3e %7.3e   %7.3e %7.3e %7.3e   %7.3e   %7.3e %7.3e %7.3e  %7.3e %7.3e %7.3e\n",
		matl[dum].thrml_cond.x, matl[dum].thrml_cond.y, matl[dum].thrml_cond.z,
		matl[dum].thrml_expn.x, matl[dum].thrml_expn.y, matl[dum].thrml_expn.z,
		matl[dum].film,
		matl[dum].E.x, matl[dum].E.y, matl[dum].E.z, 
		matl[dum].nu.xy, matl[dum].nu.xz, matl[dum].nu.yz);
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numel; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "connectivity for element (%6d) ",dum);
	   for( j = 0; j < npel; ++j )
	   {
		fscanf( o1, "%d",(connect+npel*dum+j));
		printf( "%6d ",*(connect+npel*dum+j));
	   }
	   fscanf( o1,"%d\n",(el_matl+dum));
	   printf( " with matl %3d\n",*(el_matl+dum));
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numel_film; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "convection connectivity for surface (%6d) ",dum);
	   for( j = 0; j < npel_film; ++j )
	   {
		fscanf( o1, "%d",(connect_film+npel_film*dum+j));
		printf( "%6d ",*(connect_film+npel_film*dum+j));
	   }
	   fscanf( o1,"%d\n",(el_matl_film+dum));
	   printf( " with matl %3d\n",*(el_matl_film+dum));
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < numnp; ++i )
	{
	   fscanf( o1,"%d ",&dum);
	   printf( "coordinate (%6d) ",dum);
	   printf( "coordinates ");
	   for( j = 0; j < nsd; ++j )
	   {
		fscanf( o1, "%lf ",(coord+nsd*dum+j));
		printf( "%14.6e ",*(coord+nsd*dum+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].x);
	printf( "node (%6d) has an x prescribed displacement of: ",bc.fix[dum].x);
	while( bc.fix[dum].x > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].x),
			&bc.fix[dum+1].x);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].x));
		printf( "node (%6d) has an x prescribed displacement of: ",
			bc.fix[dum+1].x);
		++dum;
	}
	bc.num_fix[0].x=dum;
	if(dum > numnp) printf( "too many prescribed displacements x\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].y);
	printf( "node (%6d) has an y prescribed displacement of: ",bc.fix[dum].y);
	while( bc.fix[dum].y > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].y+1),
			&bc.fix[dum+1].y);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].y+1));
		printf( "node (%6d) has an y prescribed displacement of: ",
			bc.fix[dum+1].y);
		++dum;
	}
	bc.num_fix[0].y=dum;
	if(dum > numnp) printf( "too many prescribed displacements y\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].z);
	printf( "node (%6d) has an z prescribed displacement of: ",bc.fix[dum].z);
	while( bc.fix[dum].z > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].z+2),
			&bc.fix[dum+1].z);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].z+2));
		printf( "node (%6d) has an z prescribed displacement of: ",
			bc.fix[dum+1].z);
		++dum;
	}
	bc.num_fix[0].z=dum;
	if(dum > numnp) printf( "too many prescribed displacements z\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.T[dum]);
	printf( "node (%6d) has a prescribed temperature of: ",bc.T[dum]);
	while( bc.T[dum] > -1 )
	{
		fscanf( o1,"%lf\n%d",(T+Tndof*bc.T[dum]),
			&bc.T[dum+1]);
		printf( "%14.6e\n",*(T+Tndof*bc.T[dum]));
		printf( "node (%6d) has a prescribed temperature of: ",
			bc.T[dum+1]);
		++dum;
	}
	bc.num_T[0]=dum;
	if(dum > numnp) printf( "too many prescribed temperatures\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.TB[dum]);
	printf( "node (%6d) has a bulk temperature of: ",bc.TB[dum]);
	while( bc.TB[dum] > -1 )
	{
		fscanf( o1,"%lf\n%d",(TB+Tndof*bc.TB[dum]),
			&bc.TB[dum+1]);
		printf( "%14.6e\n",*(TB+Tndof*bc.TB[dum]));
		printf( "node (%6d) has a bulk temperature of: ",
			bc.TB[dum+1]);
		++dum;
	}
	bc.num_TB[0]=dum;
	if(dum > numnp) printf( "too many prescribed bulk temperatures\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	printf("force vector for node: ");
	fscanf( o1,"%d",&bc.force[dum]);
	printf( "(%6d)",bc.force[dum]);
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
	   printf( "(%6d)",bc.force[dum]);
	}
	bc.num_force[0]=dum;
	if(dum > numnp) printf( "too many forces\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	printf("heat Q for node: ");
	fscanf( o1,"%d",&bc.Q[dum]);
	printf( "(%6d)",bc.Q[dum]);
	while( bc.Q[dum] > -1 )
	{
	   for( j = 0; j < Tndof; ++j )
	   {
		fscanf( o1,"%lf ",(Q+Tndof*bc.Q[dum]+j));
		printf("%14.6e ",*(Q+Tndof*bc.Q[dum]+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	   printf("heat Q for node: ");
	   ++dum;
	   fscanf( o1,"%d",&bc.Q[dum]);
	   printf( "(%6d)",bc.Q[dum]);
	}
	bc.num_Q[0]=dum;
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	printf("heat generation for node: ");
	fscanf( o1,"%d",&bc.heat_node[dum]);
	printf( "(%6d)",bc.heat_node[dum]);
	while( bc.heat_node[dum] > -1 )
	{
	   for( j = 0; j < Tndof; ++j )
	   {
		fscanf( o1,"%lf ",(heat_node+Tndof*bc.heat_node[dum]+j));
		printf("%14.6e ",*(heat_node+Tndof*bc.heat_node[dum]+j));
	   }
	   fscanf( o1,"\n");
	   printf( "\n");
	   printf("heat generation for node: ");
	   ++dum;
	   fscanf( o1,"%d",&bc.heat_node[dum]);
	   printf( "(%6d)",bc.heat_node[dum]);
	}
	bc.num_heat_node[0]=dum;
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	printf("heat generation for element: ");
	fscanf( o1,"%d",&bc.heat_el[dum]);
	printf( "(%6d)",bc.heat_el[dum]);
	while( bc.heat_el[dum] > -1 )
	{
	   fscanf( o1,"%lf ",(heat_el+bc.heat_el[dum]));
	   printf("%14.6e ",*(heat_el+bc.heat_el[dum]));
	   fscanf( o1,"\n");
	   printf( "\n");
	   printf("heat generation for element: ");
	   ++dum;
	   fscanf( o1,"%d",&bc.heat_el[dum]);
	   printf( "(%6d)",bc.heat_el[dum]);
	}
	bc.num_heat_el[0]=dum;
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	if(stress_read_flag)
	{
	   printf("stress for node: ");
	   fscanf( o1,"%d",&dum);
	   printf( "(%6d)",dum);
	   while( dum > -1 )
	   {
		fscanf( o1,"%lf ",&stress_node[dum].xx);
		fscanf( o1,"%lf ",&stress_node[dum].yy);
		fscanf( o1,"%lf ",&stress_node[dum].zz);
		fscanf( o1,"%lf ",&stress_node[dum].xy);
		fscanf( o1,"%lf ",&stress_node[dum].zx);
		fscanf( o1,"%lf ",&stress_node[dum].yz);
		printf(" %12.5e",stress_node[dum].xx);
		printf(" %12.5e",stress_node[dum].yy);
		printf(" %12.5e",stress_node[dum].zz);
		printf(" %12.5e",stress_node[dum].xy);
		printf(" %12.5e",stress_node[dum].zx);
		printf(" %12.5e",stress_node[dum].yz);
		fscanf( o1,"\n");
		printf( "\n");
		printf("stress for node: ");
		fscanf( o1,"%d",&dum);
		printf( "(%6d)",dum);
	   }
	}
	printf( "\n\n");

	if(element_stress_read_flag)
	{
	   fgets( buf, BUFSIZ, o4 );
	   printf( "\n\n");
	   printf("stress for ele: ");
	   fscanf( o4,"%d",&dum);
	   printf( "(%6d)",dum);
	   while( dum > -1 )
	   {
		fscanf( o4,"%d",&dum2);
		printf( " node (%1d)",dum2);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].xx);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].yy);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].zz);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].xy);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].zx);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].yz);
		printf(" %12.5e",stress[dum].pt[dum2].xx);
		printf(" %12.5e",stress[dum].pt[dum2].yy);
		printf(" %12.5e",stress[dum].pt[dum2].zz);
		printf(" %12.5e",stress[dum].pt[dum2].xy);
		printf(" %12.5e",stress[dum].pt[dum2].zx);
		printf(" %12.5e",stress[dum].pt[dum2].yz);
		fscanf( o4,"\n");
		printf( "\n");
		printf("stress for ele: ");
		fscanf( o4,"%d",&dum);
		printf( "(%6d)",dum);
	   }
	}
	printf( "\n\n");

	return 1;
}


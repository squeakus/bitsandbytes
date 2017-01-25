/*
    This library function reads in data for a finite element
    program which does analysis on a plate element.

		Updated 10/17/06

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

extern int dof, nmat, nmode, numel, numnp, plane_stress_flag;
extern int stress_read_flag, element_stress_read_flag, flag_3D, flag_quad_element;

int plreader( BOUND bc, int *connect, double *coord, int *el_matl, double *force,
	MATL *matl, MOMENT *moment, MDIM *moment_node, char *name, FILE *o1,
	STRESS *stress, SDIM *stress_node, double *U)
{
	int i, j, dum, dum1, dum2, dum3, dum4, dum5, name_length, file_loc;
	double fdum1, fdum2, fdum3;
	char *ccheck, one_char;
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

	printf( "number of elements:%d nodes:%d materials:%d modes:%d dof:%d\n",
		numel,numnp,nmat,nmode,dof);
	printf( "Plane Theory :%d\n", plane_stress_flag);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < nmat; ++i )
	{
	   fscanf( o1, "%d ",&dum);
	   printf( "material (%3d) Emod, nu, density, thickness, shear fac.: ",dum);
	   fscanf( o1, " %lf %lf %lf %lf %lf\n", &matl[dum].E, &matl[dum].nu,
		&matl[dum].rho, &matl[dum].thick, &matl[dum].shear);
	   printf( " %7.3e %7.3e %7.3e %7.3e %7.3e\n", matl[dum].E, matl[dum].nu,
		matl[dum].rho, matl[dum].thick, matl[dum].shear);
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	file_loc = ftell(o1);

	fscanf( o1,"%d %d %d %d %d",&dum, &dum1, &dum2, &dum3, &dum4);

/* Check if this is a quadrilateral or triangle element data file.  */

	flag_quad_element = 0;
	while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	{
		if(one_char != ' ' )
		{
		    ungetc( one_char, o1);
		    fscanf( o1,"%d", &dum5);
		    flag_quad_element = 1;
		    break;
		}
	}

	fseek(o1, file_loc, 0);

	if( !flag_quad_element )
	{

/* We have triangle elements. */

	   for( i = 0; i < numel; ++i )
	   {
		fscanf( o1,"%d ",&dum);
		printf( "connectivity for element (%6d) ",dum);
		for( j = 0; j < npel3; ++j )
		{
			fscanf( o1, "%d",(connect+npel3*dum+j));
			printf( "%6d ",*(connect+npel3*dum+j));
		}
		fscanf( o1,"%d\n",(el_matl+dum));
		printf( " with matl %3d\n",*(el_matl+dum));
	   }
	}
	else
	{
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
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	file_loc = ftell(o1);

	fscanf( o1,"%d %lf %lf",&dum, &fdum1, &fdum2);

/* Check if there is additional coordinate z data and that this is a 3-D file.  */

	flag_3D = 0;
	while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	{
		if(one_char != ' ' ) 
		{
		    ungetc( one_char, o1);
		    fscanf( o1,"%lf", &fdum3);
		    flag_3D = 1;
		    break;
		} 
	}

	fseek(o1, file_loc, 0);

	if( !flag_3D )
	{
	   for( i = 0; i < numnp; ++i )
	   {
		fscanf( o1,"%d ",&dum);
		printf( "coordinate (%6d) ",dum);
		printf( "coordinates ");
		fscanf( o1, "%lf %lf",(coord+nsd*dum), (coord+nsd*dum+1));
		printf( "%14.6e %14.6e",*(coord+nsd*dum), *(coord+nsd*dum+1));
		*(coord+nsd*dum+2) = 0.0;
		fscanf( o1,"\n");
		printf( "\n");
	   }
	}
	else
	{
	   for( i = 0; i < numnp; ++i )
	   {
		fscanf( o1,"%d ",&dum);
		printf( "coordinate (%6d) ",dum);
		printf( "coordinates ");
		fscanf( o1, "%lf %lf %lf",(coord+nsd*dum), (coord+nsd*dum+1), (coord+nsd*dum+2));
		printf( "%14.6e %14.6e %14.6e",*(coord+nsd*dum), *(coord+nsd*dum+1), *(coord+nsd*dum+2));
		fscanf( o1,"\n");
		printf( "\n");
	   }
	}
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	if( !flag_3D )
	{
	   for( i = 0; i < numnp; ++i )
	   {
		bc.fix[i].x = i;
		bc.fix[i].y = i;
		bc.fix[i].phiz = i;
		*(U+ndof6*i) = 0.0;
		*(U+ndof6*i+1) = 0.0;
		*(U+ndof6*i+5) = 0.0;
	   }
	   bc.num_fix[0].x=numnp;
	   bc.num_fix[0].y=numnp;
	   bc.num_fix[0].phiz=numnp;
	}
	else
	{
	   dum= 0;
	   fscanf( o1,"%d",&bc.fix[dum].x);
	   printf( "node (%6d) has an x prescribed displacement of: ",bc.fix[dum].x);
	   while( bc.fix[dum].x > -1 )
	   {
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].x),
			&bc.fix[dum+1].x);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].x));
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
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].y+1),
			&bc.fix[dum+1].y);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].y+1));
		printf( "node (%6d) has an y prescribed displacement of: ",
			bc.fix[dum+1].y);
		++dum;
	   }
	   bc.num_fix[0].y=dum;
	   if(dum > numnp) printf( "too many prescribed displacements y\n");
	   fscanf( o1,"\n");
	   fgets( buf, BUFSIZ, o1 );
	   printf( "\n\n");
	}

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].z);
	printf( "node (%6d) has a z prescribed displacement of: ",bc.fix[dum].z);
	while( bc.fix[dum].z > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].z+2),
			&bc.fix[dum+1].z);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].z+2));
		printf( "node (%6d) has a z prescribed displacement of: ",
			bc.fix[dum+1].z);
		++dum;
	}
	bc.num_fix[0].z=dum;
	if(dum > numnp) printf( "too many prescribed displacements z\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].phix);
	printf( "node (%6d) has a prescribed angle phi x of: ",bc.fix[dum].phix);
	while( bc.fix[dum].phix > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].phix+3),
			&bc.fix[dum+1].phix);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].phix+3));
		printf( "node (%6d) has a prescribed angle phi x of: ",
			bc.fix[dum+1].phix);
		++dum;
	}
	bc.num_fix[0].phix=dum;
	if(dum > numnp) printf( "too many prescribed angles phi x\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].phiy);
	printf( "node (%6d) has a prescribed angle phi y of: ",bc.fix[dum].phiy);
	while( bc.fix[dum].phiy > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].phiy+4),
			&bc.fix[dum+1].phiy);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].phiy+4));
		printf( "node (%6d) has a prescribed angle phi y of: ",
			bc.fix[dum+1].phiy);
		++dum;
	}
	bc.num_fix[0].phiy=dum;
	if(dum > numnp) printf( "too many prescribed angles phi y\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	if( flag_3D )
	{
	   dum= 0;
	   fscanf( o1,"%d",&bc.fix[dum].phiz);
	   printf( "node (%6d) has a prescribed angle phi z of: ",bc.fix[dum].phiz);
	   while( bc.fix[dum].phiz > -1 )
	   {
		fscanf( o1,"%lf\n%d",(U+ndof6*bc.fix[dum].phiz+5),
			&bc.fix[dum+1].phiz);
		printf( "%14.6e\n",*(U+ndof6*bc.fix[dum].phiz+5));
		printf( "node (%6d) has a prescribed angle phi z of: ",
			bc.fix[dum+1].phiz);
		++dum;
	   }
	   bc.num_fix[0].phiz=dum;
	   if(dum > numnp) printf( "too many prescribed angles phi z\n");
	   fscanf( o1,"\n");
	   fgets( buf, BUFSIZ, o1 );
	   printf( "\n\n");
	}

	dum= 0;
	printf("force vector for node: ");
	fscanf( o1,"%d",&bc.force[dum]);
	printf( "(%6d)",bc.force[dum]);
	while( bc.force[dum] > -1 )
	{
	   if( !flag_3D )
	   {
		*(force+ndof6*bc.force[dum]) = 0.0;
		*(force+ndof6*bc.force[dum]+1) = 0.0;
		fscanf( o1,"%lf %lf %lf", (force+ndof6*bc.force[dum]+2),
			(force+ndof6*bc.force[dum]+3),
			(force+ndof6*bc.force[dum]+4));
		printf("%14.6e %14.6e %14.6e", *(force+ndof6*bc.force[dum]+2),
			*(force+ndof6*bc.force[dum]+3),
			*(force+ndof6*bc.force[dum]+4));
		*(force+ndof6*bc.force[dum]+5) = 0.0;
	   }
	   else
	   {
		fscanf( o1,"%lf %lf %lf %lf %lf %lf", (force+ndof6*bc.force[dum]),
			(force+ndof6*bc.force[dum]+1),
			(force+ndof6*bc.force[dum]+2),
			(force+ndof6*bc.force[dum]+3),
			(force+ndof6*bc.force[dum]+4),
			(force+ndof6*bc.force[dum]+5));
		printf("%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e", *(force+ndof6*bc.force[dum]),
			*(force+ndof6*bc.force[dum]+1),
			*(force+ndof6*bc.force[dum]+2),
			*(force+ndof6*bc.force[dum]+3),
			*(force+ndof6*bc.force[dum]+4),
			*(force+ndof6*bc.force[dum]+5));
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

	if(stress_read_flag)
	{
	    if(!flag_3D)
	    {
		printf("stress for node: ");
		fscanf( o1,"%d",&dum);
		printf( "(%6d)",dum);
		while( dum > -1 )
		{
			fscanf( o1,"%lf ",&moment_node[dum].xx);
			fscanf( o1,"%lf ",&moment_node[dum].yy);
			fscanf( o1,"%lf ",&moment_node[dum].xy);
			fscanf( o1,"%lf ",&stress_node[dum].zx);
			fscanf( o1,"%lf ",&stress_node[dum].yz);
			printf(" %12.5e",moment_node[dum].xx);
			printf(" %12.5e",moment_node[dum].yy);
			printf(" %12.5e",moment_node[dum].xy);
			printf(" %12.5e",stress_node[dum].zx);
			printf(" %12.5e",stress_node[dum].yz);
			fscanf( o1,"\n");
			printf( "\n");
			printf("stress for node: ");
			fscanf( o1,"%d",&dum);
			printf( "(%6d)",dum);
		}
	    }
	    else
	    {
		printf("stress for node: ");
		fscanf( o1,"%d",&dum);
		printf( "(%6d)",dum);
		while( dum > -1 )
		{
			fscanf( o1,"%lf ",&moment_node[dum].xx);
			fscanf( o1,"%lf ",&moment_node[dum].yy);
			fscanf( o1,"%lf ",&moment_node[dum].xy);
			fscanf( o1,"%lf ",&stress_node[dum].xx);
			fscanf( o1,"%lf ",&stress_node[dum].yy);
			fscanf( o1,"%lf ",&stress_node[dum].xy);
			fscanf( o1,"%lf ",&stress_node[dum].zx);
			fscanf( o1,"%lf ",&stress_node[dum].yz);
			printf(" %12.5e",moment_node[dum].xx);
			printf(" %12.5e",moment_node[dum].yy);
			printf(" %12.5e",moment_node[dum].xy);
			printf(" %12.5e",stress_node[dum].xx);
			printf(" %12.5e",stress_node[dum].yy);
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
	}
	printf( "\n\n");

	if(element_stress_read_flag)
	{
	   if(!flag_3D)
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
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].xx);
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].yy);
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].xy);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].zx);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].yz);
		    printf(" %12.5e",moment[dum].pt[dum2].xx);
		    printf(" %12.5e",moment[dum].pt[dum2].yy);
		    printf(" %12.5e",moment[dum].pt[dum2].xy);
		    printf(" %12.5e",stress[dum].pt[dum2].zx);
		    printf(" %12.5e",stress[dum].pt[dum2].yz);
		    fscanf( o4,"\n");
		    printf( "\n");
		    printf("stress for ele: ");
		    fscanf( o4,"%d",&dum);
		    printf( "(%6d)",dum);
		}
	   }
	   else
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
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].xx);
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].yy);
		    fscanf( o4,"%lf ",&moment[dum].pt[dum2].xy);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].xx);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].yy);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].xy);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].zx);
		    fscanf( o4,"%lf ",&stress[dum].pt[dum2].yz);
		    printf(" %12.5e",moment[dum].pt[dum2].xx);
		    printf(" %12.5e",moment[dum].pt[dum2].yy);
		    printf(" %12.5e",moment[dum].pt[dum2].xy);
		    printf(" %12.5e",stress[dum].pt[dum2].xx);
		    printf(" %12.5e",stress[dum].pt[dum2].yy);
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
	}
	printf( "\n\n");

	return 1;
}


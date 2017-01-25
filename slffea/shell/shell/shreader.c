/*
    This library function reads in data for a finite element
    program which does analysis on a shell element.

		Updated 9/30/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shconst.h"
#include "shstruct.h"

extern int dof, nmat, nmode, numel, numnp;
extern int stress_read_flag, element_stress_read_flag, doubly_curved_flag,
	flag_quad_element;

int shreader( BOUND bc, int *connect, double *coord, int *el_matl, double *force,
	MATL *matl, char *name, FILE *o1, STRESS *stress, SDIM *stress_node,
	double *U)
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
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < nmat; ++i )
	{
	   fscanf( o1, "%d ",&dum);
	   printf( "material (%3d) Emod, nu, density: ",dum);
	   fscanf( o1, " %lf %lf %lf", &matl[dum].E, &matl[dum].nu,
		&matl[dum].rho);
	   printf( " %7.3e %7.3e %7.3e", matl[dum].E, matl[dum].nu,
		matl[dum].rho);

/* Check if there is additional material data for thickness and read the data if
   it exists.  Note that the thickness data, if it exists, should be between the
   density and the shear correction factor.
*/
	   fscanf( o1, "%lf", &fdum1);
	   while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	   {
		matl[dum].extrathick = 0;
		matl[dum].thick = 0.0;
		if(one_char != ' ' )
		{
		    ungetc( one_char, o1);
		    fscanf( o1,"%lf", &fdum2);
		    matl[dum].extrathick = 1;
		    break;
		}
	   }
	   if( matl[dum].extrathick )
	   {
		matl[dum].thick = fdum1;
		matl[dum].shear = fdum2;
		printf( " thickness, shear fac.: %7.3e %7.3e",
			matl[dum].thick, matl[dum].shear);
	   }
	   else
	   {
		matl[dum].thick = 0.0;
		matl[dum].shear = fdum1;
		printf( " shear fac.: %7.3e", matl[dum].shear);
	   }
	   fscanf( o1,"\n");
	   printf( "\n");

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
	       for( j = 0; j < npell3; ++j )
	       {
		    fscanf( o1, "%d",(connect+npell3*dum+j));
		    printf( "%6d ",*(connect+npell3*dum+j));
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
	       for( j = 0; j < npell4; ++j )
	       {
		    fscanf( o1, "%d",(connect+npell4*dum+j));
		    printf( "%6d ",*(connect+npell4*dum+j));
	       }
	       fscanf( o1,"%d\n",(el_matl+dum));
	       printf( " with matl %3d\n",*(el_matl+dum));
	    }
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

/* Store this location in the input file. We start reading again from this point
   after determing whether this is a 4 or 8 node file.
*/

	file_loc = ftell(o1);

	doubly_curved_flag = 0;
	fscanf( o1,"%d ",&dum2);
	if( dum2 > -1 )
	{

/* Check if this is a 4 node or 8 node shell data file.  If it is an 8 node file,
   then the corresponding top nodes are read in.  If it is a 4 node file, then we
   immediately begin reading prescribed displacement in x.
*/

	    fscanf( o1,"%lf", &fdum1);
	    while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	    {
	       doubly_curved_flag = 0;
	       fdum2 = 0.0;
	       fdum3 = 0.0;
	       if(one_char != ' ' )
	       {
		    ungetc( one_char, o1);
		    fscanf( o1,"%lf %lf", &fdum2, &fdum3);
		    doubly_curved_flag = 1;
		    break;
	       }
	    }
	}
	else
	{
		doubly_curved_flag = 0;
	}

	fseek(o1, file_loc, 0);

	if(doubly_curved_flag)
	{
	    printf( "The corresponding top nodes are:\n");
	    for( i = 0; i < numnp; ++i )
	    {
		fscanf( o1,"%d ",&dum);
		printf( "coordinate (%6d) ",dum);
		printf( "coordinates ");
		for( j = 0; j < nsd; ++j )
		{
		    fscanf( o1, "%lf ",(coord+nsd*(dum+numnp)+j));
		    printf( "%14.6e ",*(coord+nsd*(dum+numnp)+j));
		}
		fscanf( o1,"\n");
		printf( "\n");
	    }
	    fgets( buf, BUFSIZ, o1 );
	    printf( "\n");
	}

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].x);
	printf( "node (%6d) has a x prescribed displacement of: ",bc.fix[dum].x);
	while( bc.fix[dum].x > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].x),
			&bc.fix[dum+1].x);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].x));
		printf( "node (%6d) has a x prescribed displacement of: ",
			bc.fix[dum+1].x);
		++dum;
	}
	bc.num_fix[0].x=dum;
	if(dum > numnp) printf( "too many prescribed displacements x\n");
	fscanf( o1,"%d",&dum);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");


	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].y);
	printf( "node (%6d) has a y prescribed displacement of: ",bc.fix[dum].y);
	while( bc.fix[dum].y > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].y+1),
			&bc.fix[dum+1].y);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].y+1));
		printf( "node (%6d) has a y prescribed displacement of: ",
			bc.fix[dum+1].y);
		++dum;
	}
	bc.num_fix[0].y=dum;
	if(dum > numnp) printf( "too many prescribed displacements y\n");
	fscanf( o1,"%d",&dum);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].z);
	printf( "node (%6d) has a z prescribed displacement of: ",bc.fix[dum].z);
	while( bc.fix[dum].z > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].z+2),
			&bc.fix[dum+1].z);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].z+2));
		printf( "node (%6d) has a z prescribed displacement of: ",
			bc.fix[dum+1].z);
		++dum;
	}
	bc.num_fix[0].z=dum;
	if(dum > numnp) printf( "too many prescribed displacements z\n");
	fscanf( o1,"%d",&dum);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].phix);
	printf( "node (%6d) has a prescribed angle phi x of: ",bc.fix[dum].phix);
	while( bc.fix[dum].phix > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].phix+3),
			&bc.fix[dum+1].phix);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].phix+3));
		printf( "node (%6d) has a prescribed angle phi x of: ",
			bc.fix[dum+1].phix);
		++dum;
	}
	bc.num_fix[0].phix=dum;
	if(dum > numnp) printf( "too many prescribed angles phi x\n");
	fscanf( o1,"%d",&dum);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].phiy);
	printf( "node (%6d) has a prescribed angle phi y of: ",bc.fix[dum].phiy);
	while( bc.fix[dum].phiy > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].phiy+4),
			&bc.fix[dum+1].phiy);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].phiy+4));
		printf( "node (%6d) has a prescribed angle phi y of: ",
			bc.fix[dum+1].phiy);
		++dum;
	}
	bc.num_fix[0].phiy=dum;
	if(dum > numnp) printf( "too many prescribed angles phi y\n");
	fscanf( o1,"%d",&dum);
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
	fscanf( o1,"%d",&dum);
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
		fscanf( o1,"%lf ",&stress_node[dum].xy);
		fscanf( o1,"%lf ",&stress_node[dum].zx);
		fscanf( o1,"%lf ",&stress_node[dum].yz);
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
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].xy);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].zx);
		fscanf( o4,"%lf ",&stress[dum].pt[dum2].yz);
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
	printf( "\n\n");

	return 1;
}


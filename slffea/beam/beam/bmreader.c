/*
    This library function reads in data for a finite element
    program which does analysis on a beam.

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
#include "bmconst.h"
#include "bmstruct.h"

extern int dof, nmat, nmode, numel, numnp;
extern int stress_read_flag, stress_xyzx_flag;

int bmreader(double *axis_z, BOUND bc, int *connect, double *coord,
	double *dist_load, int *el_matl, int *el_type, double *force,
	MATL *matl, MOMENT *moment, FILE *o1, STRESS *stress,
	double *U)
{
	int i, j, dum, dum2, file_loc;
	double fdum1, fdum2, fdum3, fdum4, fdum5, fdum6;
	char *ccheck, one_char;
	char buf[ BUFSIZ ];
	char text;

	printf( "number of elements:%d nodes:%d materials:%d modes:%d dof:%d\n",
		numel,numnp,nmat,nmode,dof);
	fgets( buf, BUFSIZ, o1 );
	printf( "\n");

	for( i = 0; i < nmat; ++i )
	{
	   fscanf( o1, "%d ",&dum);
	   printf( "material (%3d) Emod, nu, density, Area, Iy, Iz",dum);
	   fscanf( o1, " %lf %lf %lf %lf %lf %lf", &matl[dum].E, &matl[dum].nu,
		&matl[dum].rho, &matl[dum].area, &matl[dum].Iy, &matl[dum].Iz);
	   printf( " %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e", matl[dum].E,
		matl[dum].nu, matl[dum].rho, matl[dum].area, matl[dum].Iy, matl[dum].Iz);

/* Check if there is additional material data for effective shear area in y and z:
   matl[dum].areaSy, matl[dum].areaSz and read the data if it exists.
*/
	   while(( one_char = (unsigned char) fgetc(o1)) != '\n')
	   {
		matl[dum].extraArea = 0;
		matl[dum].areaSy = 0.0;
		if(one_char != ' ' )
		{
		    ungetc( one_char, o1);
		    printf( " Area Sy");
		    fscanf( o1,"%lf", &matl[dum].areaSy);
		    printf( " %7.3e", matl[dum].areaSy);
		    matl[dum].extraArea = 1;
		    while(( one_char = (unsigned char) fgetc(o1)) != '\n')
		    {
			matl[dum].areaSz = 0.0;
			if(one_char != ' ' )
			{
			   ungetc( one_char, o1);
			   printf( ", Area Sz");
			   fscanf( o1,"%lf", &matl[dum].areaSz);
			   printf( " %7.3e", matl[dum].areaSz);
			   break;
			}
		    }
		    break;
		}
	   }

	   fscanf( o1,"\n");
	   printf( "\n");

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
	   fscanf( o1,"%d",(el_matl+dum));
	   printf( " with matl %3d  ",*(el_matl+dum));
	   fscanf( o1,"%d\n",(el_type+dum));
	   printf( " and type %3d\n",*(el_type+dum));
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

	printf("local z axis for element: ");
	fscanf( o1,"%d",&dum);
	printf( "(%6d)",dum);
	while( dum > -1 )
	{
	   fscanf( o1,"%lf %lf %lf\n",(axis_z + nsd*dum),
		(axis_z + nsd*dum + 1), (axis_z + nsd*dum + 2));
	   printf("%14.6e %14.6e  %14.6e\n",*(axis_z + nsd*dum),
		*(axis_z + nsd*dum + 1), *(axis_z + nsd*dum + 2));
	   ++dum;
	   printf("local z axis for element: ");
	   fscanf( o1,"%d",&dum);
	   printf( "(%6d)",dum);
	}
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

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
	fscanf( o1,"%d",&bc.fix[dum].phix);
	printf( "node (%6d) has a prescribed angle phi x of: ",bc.fix[dum].phix);
	while( bc.fix[dum].phix > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].phix+3),
			&bc.fix[dum+1].phix);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].phix+3));
		printf( "node (%6d) has an phix prescribed displacement of: ",
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
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].phiy+4),
			&bc.fix[dum+1].phiy);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].phiy+4));
		printf( "node (%6d) has an phiy prescribed displacement of: ",
			bc.fix[dum+1].phiy);
		++dum;
	}
	bc.num_fix[0].phiy=dum;
	if(dum > numnp) printf( "too many prescribed angles phi y\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	dum= 0;
	fscanf( o1,"%d",&bc.fix[dum].phiz);
	printf( "node (%6d) has a prescribed angle phi z of: ",bc.fix[dum].phiz);
	while( bc.fix[dum].phiz > -1 )
	{
		fscanf( o1,"%lf\n%d",(U+ndof*bc.fix[dum].phiz+5),
			&bc.fix[dum+1].phiz);
		printf( "%14.6e\n",*(U+ndof*bc.fix[dum].phiz+5));
		printf( "node (%6d) has an phiz prescribed displacement of: ",
			bc.fix[dum+1].phiz);
		++dum;
	}
	bc.num_fix[0].phiz=dum;
	if(dum > numnp) printf( "too many prescribed angles phi z\n");
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
	printf("distributed load for element: ");
	fscanf( o1,"%d",&bc.dist_load[dum]);
	printf( "(%6d)",bc.dist_load[dum]);
	while( bc.dist_load[dum] > -1 )
	{
	   fscanf( o1,"%lf %lf\n",(dist_load + 2*bc.dist_load[dum]),
		(dist_load + 2*bc.dist_load[dum] + 1));
	   printf(" %14.6e %14.6e\n",*(dist_load + 2*bc.dist_load[dum]),
		*(dist_load + 2*bc.dist_load[dum] + 1));
	   ++dum;
	   printf("distributed load for element: ");
	   fscanf( o1,"%d",&bc.dist_load[dum]);
	   printf( "(%6d)",bc.dist_load[dum]);
	}
	bc.num_dist_load[0]=dum;
	if(dum > numel) printf( "too many distributed loads\n");
	fscanf( o1,"\n");
	fgets( buf, BUFSIZ, o1 );
	printf( "\n\n");

	if(stress_read_flag)
	{
/* Store this location in the input file. We start reading again from this point
   after determing whether this is a new or old beam file.
*/
	   file_loc = ftell(o1);

	   fscanf( o1,"%d",&dum);
	   if( dum > -1 )
	   {

/* Check if this is a new beam file where there is stress xy and stress zx data
   and read the data if it exists.  If not, stress_xyzx_flag = 0 and we are reading
   an old beam data file.
*/

		fscanf( o1,"%d",&dum2);

		fscanf( o1,"%lf %lf %lf %lf", &fdum1, &fdum2, &fdum3, &fdum4);
		while(( one_char = (unsigned char) fgetc(o1)) != '\n')
		{
		   stress_xyzx_flag = 0;
		   fdum5 = 0.0;
		   fdum6 = 0.0;
		   if(one_char != ' ' )
		   {
			ungetc( one_char, o1);
			fscanf( o1,"%lf %lf", &fdum5, &fdum6);
			stress_xyzx_flag = 1;
			break;
		   }
		}

	   }

	   fseek(o1, file_loc, 0);

	   printf("stress for ele: ");
	   fscanf( o1,"%d",&dum);
	   printf( "(%6d)",dum);
	   if(stress_xyzx_flag)
	   {
		while( dum > -1 )
		{
		   fscanf( o1,"%d",&dum2);
		   printf( " node (%1d)",dum2);
		   fscanf( o1,"%lf ",&stress[dum].pt[dum2].xx);
		   fscanf( o1,"%lf ",&stress[dum].pt[dum2].xy);
		   fscanf( o1,"%lf ",&stress[dum].pt[dum2].zx);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].xx);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].yy);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].zz);
		   printf(" %12.5e",stress[dum].pt[dum2].xx);
		   printf(" %12.5e",stress[dum].pt[dum2].xy);
		   printf(" %12.5e",stress[dum].pt[dum2].zx);
		   printf(" %12.5e",moment[dum].pt[dum2].xx);
		   printf(" %12.5e",moment[dum].pt[dum2].yy);
		   printf(" %12.5e",moment[dum].pt[dum2].zz);
		   fscanf( o1,"\n");
		   printf( "\n");
		   printf("stress for ele: ");
		   fscanf( o1,"%d",&dum);
		   printf( "(%6d)",dum);
		}
	   }
	   else
	   {
		while( dum > -1 )
		{
		   fscanf( o1,"%d",&dum2);
		   printf( " node (%1d)",dum2);
		   fscanf( o1,"%lf ",&stress[dum].pt[dum2].xx);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].xx);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].yy);
		   fscanf( o1,"%lf ",&moment[dum].pt[dum2].zz);
		   printf(" %12.5e",stress[dum].pt[dum2].xx);
		   printf(" %12.5e",stress[dum].pt[dum2].xy);
		   printf(" %12.5e",stress[dum].pt[dum2].zx);
		   printf(" %12.5e",moment[dum].pt[dum2].xx);
		   printf(" %12.5e",moment[dum].pt[dum2].yy);
		   printf(" %12.5e",moment[dum].pt[dum2].zz);
		   fscanf( o1,"\n");
		   printf( "\n");
		   printf("stress for ele: ");
		   fscanf( o1,"%d",&dum);
		   printf( "(%6d)",dum);
	   	}
	   }
	}

	printf( "\n\n");

	return 1;
}


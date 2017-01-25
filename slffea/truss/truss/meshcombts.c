/*
    This program generates the mesh for a honeycomb
    using truss elements.  There are also the 2 mesh generators
    based on this code, "meshcomb-flat.c" for a flat honeycomb,
    and "meshcombts.c" for trusses that form a hexagon structure.

    Because of the nature of truss elemets, it is very difficult
    to create a mesh that doesn't have rigid body movement due
    to its lack of being able to handle bending.  Most of the
    mesh generated should be for beam elements.

    Even adding cross elements in each comb, I have to fix all
    displacements in z.  There may not be a senario that will
    work with the curved comb.

		Updated 12/29/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.

Special note
------------
Read the notes file in the directory:

   ~/slffea-1.5/quad/quad/

 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define pi              3.141592654

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, plane_stress_flag, comb_backing_flag, flat_comb_flag;
	int i,j,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, fdum4, ang60, ang30;
	int horiz_el_num, vert_el_num, center_shift;
	double comb_R, comb_thick, horiz_length, el_length, rad_length,
		angA, angB, ang1, ang2, ang3, ang4, arc_angle, A, B, X, Y, Z;
	int side_el_num, angle_div, angleh_div;
	char name[20];
	char buf[ BUFSIZ ];
	FILE *o1, *o2;
	char text;

	o2 = fopen( "comb.in", "r" );
	if(o2 == NULL ) {
		printf("Can't find file comb.in\n");
		exit(1);
	}

	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&horiz_length);
	comb_R = horiz_length;
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&comb_thick);
	printf( "comb thickness: %16.8e\n ", comb_thick);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&horiz_el_num);
	printf( "Number of combs on the horizontal side: %4d\n ", horiz_el_num);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&vert_el_num);
	printf( "Number of combs on the vertical side: %4d\n ", vert_el_num);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&comb_backing_flag);
	printf( "Combs have an element backing them: ");
	if( comb_backing_flag) printf( "yes\n");
	else printf( "no\n");
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%lf\n",&arc_angle);
	printf( "arc angle: %16.8e\n ", arc_angle);
	fgets( buf, BUFSIZ, o2 );
	fscanf( o2, "%d\n",&flat_comb_flag);
	if( flat_comb_flag)
	{
		printf( "flat comb\n");
		printf( "length of horizontal side: %16.8e\n", horiz_length);
		o1 = fopen( "comb-flat","w" );
	}
	else
	{
		printf( "cylindrical comb\n");
		printf( "radius of cylinder: %16.8e\n", comb_R);
		o1 = fopen( "comb-cyl","w" );
	}

	nmat = 1; 

	plane_stress_flag = 1;
	numel = vert_el_num*(4*horiz_el_num + 2*horiz_el_num - 1) + horiz_el_num-1;
	if( comb_backing_flag) numel +=
		(vert_el_num - 1)*(2*horiz_el_num-1) + vert_el_num*(2*horiz_el_num-1) - 1;

/* Note that some adjustment may need to be made to numel since it may be off by 1 */

	numnp = (2*vert_el_num + 1)*2*horiz_el_num;

	ang60 = 60.0*pi/180;
	ang30 = 30.0*pi/180;
	arc_angle = arc_angle*pi/180.0;
	el_length = horiz_length/((double)horiz_el_num+1);
	if( !flat_comb_flag)
	{
		horiz_length = arc_angle*comb_R/3.0;
		el_length = horiz_length/((double)horiz_el_num-1);
	}
	rad_length = el_length/(2.0*sin(ang30)); /* This applies to the center of a
	                                            comb to one corner. As expected, it is the
	                                            same as el_length.  */

	printf( "   %9.6e %9.6e\n", el_length, rad_length);

	if( flat_comb_flag)
	{
	    fprintf( o1, "   numel numnp nmat nmode");
	    fprintf( o1, "  This is the flat comb mesh \n ");
	    fprintf( o1, "    %4d %4d %4d %4d \n ",
		numel,numnp,nmat,0);
	}
	else
	{
	    fprintf( o1, "   numel numnp nmat nmode");
	    fprintf( o1, "  This is the cylindrical comb mesh\n");
	    fprintf( o1, "    %4d %4d %4d %4d\n ",
		numel,numnp,nmat,0);
	}

	fprintf( o1, "matl no., E modulus, density, Area\n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o1, "%3d ",0);
	   fprintf( o1, " %9.4f %9.4f     %9.4f \n ",10000000.0, 2.77e3, 1.0);
	}

	dum = 0;
	fprintf( o1, "el no., connectivity, matl no. \n");
	for( i = 0; i < vert_el_num; ++i )
	{
		for( j = 0; j < 2*horiz_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j,
				i*4*horiz_el_num + j + 2*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < horiz_el_num-1; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + 2*j + 1,
				i*4*horiz_el_num + 2*j + 2,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < horiz_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + 2*j + 2*horiz_el_num,
				i*4*horiz_el_num + 2*j + 1 + 2*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < 2*horiz_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j + 2*horiz_el_num,
				i*4*horiz_el_num + j + 4*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	}

	i = vert_el_num;
	for( j = 0; j < horiz_el_num-1; ++j )
	{
		fprintf( o1, "%4d %4d %4d %4d",dum,
			i*4*horiz_el_num + 2*j + 1,
			i*4*horiz_el_num + 2*j + 2,
			0);
		fprintf( o1, "\n");
		++dum;
	}


/* The code below represents the elements that are on the back of the combs. */

	if( comb_backing_flag)
	{
	    for( i = 0; i < vert_el_num; ++i )
	    {
		for( j = 1; j < 2*horiz_el_num-1; j += 2 )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j,
				i*4*horiz_el_num + j + 1 + 4*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j + 1,
				i*4*horiz_el_num + j + 4*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	    for( i = 0; i < vert_el_num - 1; ++i )
	    {
		for( j = 0; j < 2*horiz_el_num-1; j += 2 )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j + 2*horiz_el_num,
				i*4*horiz_el_num + j + 1 + 4*horiz_el_num + 2*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*4*horiz_el_num + j + 1 + 2*horiz_el_num,
				i*4*horiz_el_num + j + 4*horiz_el_num + 2*horiz_el_num,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	}


	fprintf( o1, "node no., coordinates \n");
	if(flat_comb_flag)  /* These are the nodes for the flat comb. */
	{
	    dum = 0;
	    fdum3 = 0.0;
	    fdum4 = 0.0;
	    A = 2.0*rad_length;
	    B = el_length;
	    for( i = 0; i < 2*vert_el_num + 1; ++i )
	    {
		fdum1 = 0.0;
		for( j = 0; j < horiz_el_num; ++j )  /* Nodes on inner layer */
		{
			fdum2 = fdum3;
			Y = fdum3;
			X = fdum1 + fdum4;
			Z = 0.0;
			fprintf( o1, "%d ",dum);
			fprintf( o1, "%9.4f   %9.4f  %9.4f ", X, Y, Z);
			fprintf( o1, "\n");
			++dum;
			fdum1 += A;
			fdum2 = fdum3;
			Y = fdum3;
			X = fdum1 + fdum4;
			Z = 0.0;
			fprintf( o1, "%d ",dum);
			fprintf( o1, "%9.4f   %9.4f  %9.4f ", X, Y, Z);
			fprintf( o1, "\n");
			++dum;
			fdum1 += B;
		}
		if(fdum4 < 1.0e-5)
		{
			fdum4 = (2.0*rad_length - el_length)/2.0;
			A = el_length;
			B = 2.0*rad_length;
		}
		else
		{
			fdum4 = 0.0;
			A = 2.0*rad_length;
			B = el_length;
		}
		fdum3 += el_length*cos(ang30);
	    }
	}
	else  /* These are the nodes for the cylindrical comb. */
	{
	    dum = 0;
	    fdum3 = 0.0;
	    fdum4 = 0.0;
	    A = 2.0*rad_length;
	    B = el_length;
	    ang4 = 0.0;
	    angA = 2.0*rad_length/comb_R;
	    angB = el_length/comb_R;
	    for( i = 0; i < 2*vert_el_num + 1; ++i )
	    {
		fdum1 = 0.0;
		ang1 = 0.0;
		for( j = 0; j < horiz_el_num; ++j )
		{
			Y = fdum3;
			X = comb_R*sin(ang1 + ang4);
			Z = comb_R*cos(ang1 + ang4);
			fprintf( o1, "%d ",dum);
			fprintf( o1, "%9.4f   %9.4f  %9.4f ", X, Y, Z);
			fprintf( o1, "\n");
			++dum;
			fdum1 += A;
			Y = fdum3;
			ang1 += angA;
			X = comb_R*sin(ang1 + ang4);
			Z = comb_R*cos(ang1 + ang4);
			fprintf( o1, "%d ",dum);
			fprintf( o1, "%9.4f   %9.4f  %9.4f ", X, Y, Z);
			fprintf( o1, "\n");
			++dum;
			fdum1 += B;
			ang1 += angB;
		}
		if(fdum4 < 1.0e-5)
		{
			fdum4 = (2.0*rad_length - el_length)/2.0;
			A = el_length;
			B = 2.0*rad_length;
			ang4 = (2.0*rad_length - el_length)/2.0/comb_R;
			angA = el_length/comb_R;
			angB = 2.0*rad_length/comb_R;
		}
		else
		{
			fdum4 = 0.0;
			A = 2.0*rad_length;
			B = el_length;
			ang4 = 0.0;
			angA = 2.0*rad_length/comb_R;
			angB = el_length/comb_R;
		}
		fdum3 += el_length*cos(ang30);
		ang3 = el_length*cos(ang30)/comb_R;
	    }
	}

	if(flat_comb_flag)  /* These are the nodes for the flat comb. */
	{
	    dum= 0;
	    fprintf( o1, "prescribed displacement x: node  disp value\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);

/* Endig vertical side */
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Top nodes */
	    {
		dum = 2*vert_el_num*2*horiz_el_num + i;
		fprintf( o1, "%4d %14.6e\n", dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement y: node  disp value\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);

/* Endig vertical side */
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Top nodes */
	    {
		dum = 2*vert_el_num*2*horiz_el_num + i;
		fprintf( o1, "%4d %14.6e\n", dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement z: node  disp value\n");
#if 0
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);

/* Endig vertical side */
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Top nodes */
	    {
		dum = 2*vert_el_num*2*horiz_el_num + i;
		fprintf( o1, "%4d %14.6e\n", dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
#endif

	    for( i = 0; i < numnp; ++i )
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);

	    center_shift = 4;
	    fprintf( o1, "node with point load and load vector in x, y, z\n");
	    for( i = 0; i < center_shift + 1; ++i )
	    {
		for( j = 0; j < center_shift; ++j )
		{
		    dum = (i + vert_el_num - center_shift/2)*2*horiz_el_num +
			horiz_el_num - center_shift/2 + j;
		    fprintf( o1, "%4d %14.6e %14.6e %14.6e\n", dum, 0.0, 100000.0, 0.0);
		}
	    }
	    fprintf( o1, "%4d\n ",-10);
	}
	else  /* These are the nodes for the cylindrical comb. */
	{
	    dum= 0;
	    fprintf( o1, "prescribed displacement x: node  disp value\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);

/* Endig vertical side */
#if 0
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
#endif
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement y: node  disp value\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
#if 0
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);

/* Endig vertical side */
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
#endif
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement z: node  disp value\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )      /* Bottom nodes */
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    for( i = 0; i < 2*vert_el_num + 1; ++i )   /* Side nodes */
	    {
#if 0
/* Starting vertical side */
		dum = i*2*horiz_el_num;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
#endif

/* Endig vertical side */
		dum = i*2*horiz_el_num + 2*horiz_el_num - 1;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);

/*
	    for( i = 0; i < numnp; ++i )
	    {
		dum = i;
		fprintf( o1, "%4d %14.6e\n",dum, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
*/

	    fprintf( o1, "node with point load and load vector in x, y, z\n");
	    for( i = 0; i < 2*horiz_el_num; ++i )
	    {
		dum = 2*vert_el_num*2*horiz_el_num + i;
		fprintf( o1, "%4d %14.6e %14.6e %14.6e\n", dum, 0.0, 100000.0, 0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	}

	fprintf( o1, "element and gauss pt. with stress and stress vector in xx,yy,xy\n");
	fprintf( o1, "%4d ",-10);

	return 1;
}


/*
    This program generates the mesh for the building
    problem for a beam or truss.

		Updated 12/15/08

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999-2008  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv)
{
	int dof, nmat, numel, numnp, nmode, plane_stress_flag;
	int i,j,k,dum,dum2,dum3,dum4;
	double fdum1, fdum2, fdum3, ux[9], uy[9];
	double height_length, x_length, z_length,
		height_el_length, x_el_length, z_el_length;
	int height_el_num, x_el_num, z_el_num,  nodes_per_layer, el_type;
	char buf[ BUFSIZ ];
	FILE *o1;
	char text;

	o1 = fopen( "build","w" );
	printf( "\n What is the length of the height?\n ");
	scanf( "%lf",&height_length);
	printf( "\n How many elements on the height?\n ");
	scanf( "%d",&height_el_num);
	printf( "\n What is the length of the width along x?\n ");
	scanf( "%lf",&x_length);
	printf( "\n How many elements on the width along x?\n ");
	scanf( "%d",&x_el_num);
	printf( "\n What is the length of the width along z?\n ");
	scanf( "%lf",&z_length);
	printf( "\n How many elements on the width along z?\n ");
	scanf( "%d",&z_el_num);
	printf( "\n What is the element type(1 = truss, 2 = beam?\n ");
	scanf( "%d",&el_type);

	nmode = 0; 
	nmat = 1; 

	if(el_type == 2)
		numel = height_el_num*((x_el_num+1)*(2*z_el_num + 1) + x_el_num*(z_el_num + 1));
	else
		numel = height_el_num*((x_el_num+1)*(2*z_el_num + 1) + x_el_num*(z_el_num + 1))+
		    4*(height_el_num-1)*z_el_num*x_el_num;

	numnp = (height_el_num+1)*(x_el_num+1)*(z_el_num+1);
	height_el_length = height_length/((double)height_el_num+1);
	x_el_length = x_length/((double)x_el_num+1);
	z_el_length = z_length/((double)z_el_num+1);
	nodes_per_layer = (z_el_num+1)*(x_el_num+1);
	printf( "\n %4d %4d %4d %4d\n ",(height_el_num),(x_el_num),(z_el_num),nodes_per_layer);
	printf( "\n %14.6e %14.6e %14.6e\n ",height_el_length, x_el_length, z_el_length);

	fprintf( o1, "   numel numnp nmat nmode  (This is for a beam build)\n ");
	fprintf( o1, "    %4d %4d %4d %4d \n ", numel,numnp,nmat,nmode);



	if(el_type == 2)
	{

/* For beam elements */

	  fprintf( o1, "matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n");
	  for( i = 0; i < nmat; ++i )
	  {
		fprintf( o1, "%3d ",0);
		fprintf( o1, " %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",
		    30000000.0, 0.3, 1.23e4, 2.0, 1.0, 1.0);
	  }

/* Make connectivity for building */

	  dum = 0;
	  fprintf( o1, "el no., connectivity, matl no., element type\n");

	  for( k = 0; k < height_el_num; ++k )
	  {
	    for( i = 0; i < x_el_num+1; ++i )
	    {
		for( j = 0; j < z_el_num+1; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j + k*nodes_per_layer,
				i*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				0,2);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				i*(z_el_num+1) + j + 1 + nodes_per_layer + k*nodes_per_layer,
				0,2);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	    for( i = 0; i < x_el_num; ++i )
	    {
		for( j = 0; j < z_el_num+1; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j  + nodes_per_layer + k*nodes_per_layer,
				i*(z_el_num+1) + j  + (z_el_num+1) + nodes_per_layer + k*nodes_per_layer,
				0,2);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	  }
	}
	else
	{
/* For truss elements */

	  fprintf( o1, "matl no., E mod, density, Area\n");
	  for( i = 0; i < nmat; ++i )
	  {
		fprintf( o1, "%3d ",0);
		fprintf( o1, " %9.4f %9.4f %9.4f\n",
		    30000000.0, 1.23e4, 2.0);
	  }

/* Make connectivity for building */

	  dum = 0;
	  fprintf( o1, "el no., connectivity, matl no.\n");

	  for( k = 0; k < height_el_num; ++k )
	  {
	    for( i = 0; i < x_el_num+1; ++i )
	    {
		for( j = 0; j < z_el_num+1; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j + k*nodes_per_layer,
				i*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				i*(z_el_num+1) + j + 1 + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	    for( i = 0; i < x_el_num; ++i )
	    {
		for( j = 0; j < z_el_num+1; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j  + nodes_per_layer + k*nodes_per_layer,
				i*(z_el_num+1) + j  + (z_el_num+1) + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	  }
	  for( k = 1; k < height_el_num; ++k )
	  {
	    for( i = 0; i < x_el_num; ++i )
	    {
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j + k*nodes_per_layer,
				(i+1)*(z_el_num+1) + j+1 + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				i*(z_el_num+1) + j+1 + k*nodes_per_layer,
				(i+1)*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				(i+1)*(z_el_num+1) + j+1 + k*nodes_per_layer,
				i*(z_el_num+1) + j + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
		for( j = 0; j < z_el_num; ++j )
		{
			fprintf( o1, "%4d %4d %4d %4d",dum,
				(i+1)*(z_el_num+1) + j + k*nodes_per_layer,
				i*(z_el_num+1) + j+1 + nodes_per_layer + k*nodes_per_layer,
				0);
			fprintf( o1, "\n");
			++dum;
		}
	    }
	  }
	}


	dum = 0;
	fdum1 = 0.0;
	fprintf( o1, "node no., coordinates \n");
	for( k = 0; k < height_el_num+1; ++k )
	{
	    fdum2 = 0.0;
	    for( i = 0; i < x_el_num+1; ++i )
	    {
		fdum3 = 0.0;
		for( j = 0; j < z_el_num+1; ++j )
		{
			fprintf( o1, "%d ",dum);
			fprintf( o1, "%9.4f %9.4f %9.4f ",fdum2, fdum1, fdum3);
			fprintf( o1, "\n");
			++dum;
			fdum3 += z_el_length;
		}
		fdum2 += x_el_length;
	    }
	    fdum1 += height_el_length;
	}

#if 1
	if(el_type == 2)
	{

/* For beam elements */
	    dum= 0;
	    fprintf( o1, "element with specified local z axis: x, y, z component\n");
	    for( k = 0; k < height_el_num; ++k )
	    {
		for( i = 0; i < x_el_num+1; ++i )
		{
		    for( j = 0; j < z_el_num+1; ++j )
		    {
			    fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,0.0,1.0);
			    ++dum;
		    }
		    for( j = 0; j < z_el_num; ++j )
		    {
			    fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,1.0,0.0,0.0);
			    ++dum;
		    }
		}
		for( i = 0; i < x_el_num; ++i )
		{
		    for( j = 0; j < z_el_num+1; ++j )
		    {
			    fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum,0.0,1.0,0.0);
			    ++dum;
		    }
		}
	    }
	    fprintf( o1, "%4d\n ",-10);


	    fprintf( o1, "prescribed displacement x: node  disp value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement y: node  disp value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement z: node  disp value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);

	    fprintf( o1, "prescribed angle phi x: node angle value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed angle phi y: node angle value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed angle phi z: node angle value\n");
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	}
	else
	{
/* For truss elements */

	    fprintf( o1, "prescribed displacement x: node  disp value\n");
	    for( i = nodes_per_layer; i < 2*nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement y: node  disp value\n");
	    for( i = nodes_per_layer; i < 2*nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "prescribed displacement z: node  disp value\n");
	    for( i = nodes_per_layer; i < 2*nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e\n",i,0.0);
	    }
	    fprintf( o1, "%4d\n ",-10);
	}

	if(el_type == 2)
	{
/* For beam elements */
	    fprintf( o1, "node with point load x, y, z and 3 moments phi x, phi y, phi z\n");

	    dum = nodes_per_layer*(height_el_num);
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",dum,
			100000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		++dum;
	    }
	    fprintf( o1, "%4d\n ",-10);
	    fprintf( o1, "element with distributed load in local beam y and z coordinates\n");
	    fprintf( o1, "%4d \n",-10);
	}
	else
	{
/* For truss elements */
	    fprintf( o1, "node with point load x, y, z\n");

	    dum = nodes_per_layer*(height_el_num);
	    for( i = 0; i < nodes_per_layer; ++i )
	    {
		fprintf( o1, "%4d %14.6e %14.6e %14.6e\n",dum, 100000.0, 0.0, 0.0);
		++dum;
	    }
	    fprintf( o1, "%4d\n ",-10);
	}

	fprintf( o1, "element no. and gauss pt. no. with stress and tensile stress vector\n");
	fprintf( o1, "%4d \n",-10);
#endif

	return 1;
}


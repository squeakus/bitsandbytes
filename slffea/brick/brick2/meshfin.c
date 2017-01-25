/*
    This program generates the mesh for a fined cylinder for
    either brick or brick 2 elements.
  
     Updated 10/22/01

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define pi   		3.141592654	

typedef struct {
        double xy,yx,zx,xz,yz,zy;
} ORTHO;

typedef struct {
        double x, y, z;
} XYZF;

typedef struct {
        double film;
        XYZF E;
        ORTHO nu;
        double rho;
        XYZF thrml_cond;
        XYZF thrml_expn;
} MATL;

int writer(void)
{
        FILE *o1;
        int i, i2, j, num_hor, num_ver, num_fin, num_ver_fin, num_ver_div,
		numnp, numel, numel_film, numnp_layer, numnp_sec, el_type,
		dum, dum2, dum3, dum4;
	double S, C, radius, thick, thick_fin, length_fin, X, Y, Z, length,
		length_inc, length_inc_fin, angle, angle_T, angle_T_rad, angle_inc,
		angle_inc_cyl, angle_start, angle_sec;
	double arc_length, arc_length_cyl, arc_length_fin, angle_inc_fin, fdum, fdum2;
	char buf[ BUFSIZ ];
	MATL matl;
	double heat_node, TB;

        o1 = fopen( "fin.in","r" );

        if(o1 == NULL ) {
                printf("Can't find file fin.in\n");
                exit(1);
        }

    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&num_hor);
	printf("Number of elements in horizontal\n");
    	printf("%d\n",num_hor);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&num_ver);
	printf("Number of elements in vertical\n");
    	printf("%d\n",num_ver);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&radius);
	printf("Radius\n");
    	printf("%lf\n",radius);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&angle_T);
	printf("Angle\n");
    	printf("%lf\n",angle_T);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&length);
	printf("Length\n");
    	printf("%lf\n",length);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&thick);
	printf("Thickness\n");
    	printf("%lf\n",thick);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&num_fin);
	printf("Number of fins\n");
    	printf("%d\n",num_fin);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&num_ver_fin);
	printf("Number of elements in fin vertical\n");
    	printf("%d\n",num_ver_fin);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&thick_fin);
	printf("Fin Thickness\n");
    	printf("%lf\n",thick_fin);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&length_fin);
	printf("Fin Height\n");
    	printf("%lf\n",length_fin);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%d\n",&el_type);
	printf("Element type\n");
	if(el_type == 1) printf("brick\n");
	if(el_type == 2) printf("brick 2(thermal)\n");
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&matl.E.x);
	printf("Elastic Modulus (N/mm*mm)\n");
    	printf("%lf\n",matl.E.x);
    	fgets( buf, BUFSIZ, o1 );
    	fscanf(o1, "%lf\n",&matl.nu.xy);
	printf("Poisson\n");
    	printf("%lf\n",matl.nu.xy);

	if(el_type > 1)
	{
   	 	fgets( buf, BUFSIZ, o1 );
    		fscanf(o1, "%lf\n",&matl.thrml_cond.x);
		printf("thermal conductivity W/(mm*C)\n");
    		printf("%lf\n",matl.thrml_cond.x);
    		fgets( buf, BUFSIZ, o1 );
    		fscanf(o1, "%lf\n",&matl.thrml_expn.x);
		printf("thermal coefficient of expansion (1/C)\n");
    		printf("%lf\n",matl.thrml_expn.x);
    		fgets( buf, BUFSIZ, o1 );
    		fscanf(o1, "%lf\n",&matl.film);
		printf("film coefficient for convection (W/(mm*mm*C)\n");
    		printf("%lf\n",matl.film);
    		fgets( buf, BUFSIZ, o1 );
    		fscanf(o1, "%lf\n",&heat_node);
		printf("Heat generation at nodes W/(mm*mm*mm)\n");
    		printf("%lf\n",heat_node);
    		fgets( buf, BUFSIZ, o1 );
    		fscanf(o1, "%lf\n",&TB);
		printf("Bulk Temperature\n");
    		printf("%lf\n",TB);
	}

	angle_start=90.0*pi/180.0;
	length_inc=length/(double)num_ver;
	length_inc_fin=length_fin/(double)num_ver_fin;

	angle_T_rad=angle_T*pi/180;
	arc_length = radius*angle_T_rad;
	arc_length_fin = ((double)num_fin)*thick_fin;

	arc_length_cyl = arc_length - arc_length_fin;

	if(arc_length < 0.0)
	{
		printf("\nYou must increase the radius or\n");
		printf("decrease the thickness of fins.\n");
                exit(1);
	}

	fdum = ((double)num_ver)/((double)num_fin);
	if(fdum < 1.0)
	{
		printf("\nYou must increase the number of vertical elments or\n");
		printf("decrease the number of fins.\n");
                exit(1);
	}

	num_ver_div = (int)fdum;  /* This tells how many vertical elements are spaced
                                     between the fins */

	numnp_sec = 2*num_ver_fin + 2*(num_ver_div+1) - 2;
	numnp_layer = num_fin*numnp_sec+2;

	numnp=(numnp_layer)*(num_hor+1);
	numel=(num_ver_fin + num_ver_div)*num_fin*num_hor;
	numel_film= 2*num_fin*(num_ver_fin - 1)*num_hor;

       	/*printf("%d %d %d\n",numnp_sec,numnp_layer,num_hor+1);*/


	angle_inc_cyl = arc_length_cyl/radius/(double)(num_ver_div*num_fin);
	angle_inc_fin = arc_length_fin/radius/(double)(num_fin);

	if(el_type < 2 ) o1 = fopen( "fins","w" );
	if(el_type > 1 ) o1 = fopen( "fins.th","w" );

	if(el_type < 2)
	{	
		matl.rho = 2.3e9;
        	fprintf(o1, "   numel numnp nmat nmode  (This is for a brick fin)\n");
        	fprintf(o1, "%d %d %d %d\n",numel,numnp,1,0);
        	fprintf(o1, "matl no., E modulus, Poisson Ratio, density\n");
           	fprintf( o1, "%4d   %8.2e  %8.2e  %8.2e\n ", 0, matl.E.x, matl.nu.xy, matl.rho);
	}
	if(el_type > 1)
	{	
        	fprintf(o1, "   numel numnp nmat numel_film disp_analysis thermal_analysis\n");
        	fprintf(o1, "%d %d %d %d %d %d\n",numel,numnp,1,numel_film,1,1);
		fprintf( o1, "matl no., thermal cond x, y, z, thermal expan x, y, z, ");
        	fprintf( o1, "film coeff, E modulus x, y, z, and Poisson Ratio xy, xz, xz\n");
		fprintf( o1, "%4d ",0);
           	fprintf( o1, "     %7.4e %7.4e %7.4e  %7.4e %7.4e %7.4e",
                	matl.thrml_cond.x, matl.thrml_cond.x, matl.thrml_cond.x,
                	matl.thrml_expn.x, matl.thrml_expn.x, matl.thrml_expn.x);
           	fprintf( o1, "  %7.4e  %8.2f %8.2f %8.2f  %4.2f %4.2f %4.2f\n ",
                	matl.film,
                	matl.E.x, matl.E.x, matl.E.x,
                	matl.nu.xy, matl.nu.xy, matl.nu.xy);
	}

        fprintf(o1, "el no.,connectivity, matl no.\n");
	dum = 0;
        for( i = 0; i < num_hor; ++i )
        {
            for( i2 = 0; i2 < num_fin; ++i2 )
            {
		dum3 = numnp_layer*i + numnp_sec*i2;
		dum4 = numnp_layer*(i+1) + numnp_sec*i2;
            	for( j = 0; j < num_ver_fin; ++j )
            	{
        	    fprintf(o1, "%4d ",dum);
        	    fprintf(o1, "%4d %4d %4d %4d ",
		   	dum3 + j,
			dum3 + j + 1,
			dum3 + j + 1 + num_ver_fin + 1,
			dum3 + j + num_ver_fin + 1,0);
        	    fprintf(o1, "%4d %4d %4d %4d %4d\n",
		   	dum4 + j,
			dum4 + j + 1,
			dum4 + j + 1 + num_ver_fin + 1,
			dum4 + j + num_ver_fin + 1,0);
		    ++dum;
	        }
		dum2 = 0;
		if(i2 == num_fin - 1) dum2 = 1;
        	fprintf(o1, "%4d ",dum);
        	fprintf(o1, "%4d %4d %4d %4d ",
		    dum3 + num_ver_fin + 1,
		    dum3 + num_ver_fin + 2,
		    dum3 + 2*(num_ver_fin + 1) + num_ver_div - 1 + dum2,
		    dum3 + 2*(num_ver_fin + 1),0);
        	fprintf(o1, "%4d %4d %4d %4d %4d\n",
		    dum4 + num_ver_fin + 1,
		    dum4 + num_ver_fin + 2,
		    dum4 + 2*(num_ver_fin + 1) + num_ver_div - 1 + dum2,
		    dum4 + 2*(num_ver_fin + 1),0);
		++dum;
            	for( j = 0; j < num_ver_div-2; ++j )
            	{
        	    fprintf(o1, "%4d ",dum);
        	    fprintf(o1, "%4d %4d %4d %4d ",
		   	dum3 + 2*(num_ver_fin + 1) + j,
			dum3 + 2*(num_ver_fin + 1) + j + num_ver_div - 1 + dum2,
			dum3 + 2*(num_ver_fin + 1) + j + num_ver_div + dum2,
			dum3 + 2*(num_ver_fin + 1) + j + 1,0);
        	    fprintf(o1, "%4d %4d %4d %4d %4d\n",
		   	dum4 + 2*(num_ver_fin + 1) + j,
			dum4 + 2*(num_ver_fin + 1) + j + num_ver_div - 1 + dum2,
			dum4 + 2*(num_ver_fin + 1) + j + num_ver_div + dum2,
			dum4 + 2*(num_ver_fin + 1) + j + 1,0);
		    ++dum;
	    	}
		if( i2 == num_fin - 1)
		{
        	    fprintf(o1, "%4d ",dum);
        	    fprintf(o1, "%4d %4d %4d %4d ",
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + 1,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + 1,0);
        	    fprintf(o1, "%4d %4d %4d %4d %4d\n",
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + 1,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + 1,0);
		    ++dum;
		}
		else
            	{
        	    fprintf(o1, "%4d ",dum);
        	    fprintf(o1, "%4d %4d %4d %4d ",
		   	dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div - 1 + dum2,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + 1 + dum2,
			dum3 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + dum2,0);
        	    fprintf(o1, "%4d %4d %4d %4d %4d\n",
		   	dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div - 1 + dum2,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + 1 + dum2,
			dum4 + 2*(num_ver_fin + 1) + num_ver_div - 2 + num_ver_div + dum2,0);
		    ++dum;
	    	}
	    }
	}

	if(el_type > 1)
        {
            fprintf(o1, "connectivity for convection surfaces: ");
            fprintf(o1, "surface number  connectivity, matl no.\n");
	    dum = 0;
            for( i = 0; i < num_hor; ++i )
            {
                for( i2 = 0; i2 < num_fin; ++i2 )
                {
		    dum3 = numnp_layer*i + numnp_sec*i2;
		    dum4 = numnp_layer*(i+1) + numnp_sec*i2;
            	    for( j = 1; j < num_ver_fin; ++j )
            	    {
        	    	fprintf(o1, "%4d ",dum);
        	    	fprintf(o1, "%4d %4d %4d %4d %4d\n",
		   	    dum3 + j,
			    dum3 + j + 1,
			    dum4 + j + 1,
		   	    dum4 + j,0);
		    	++dum;
        	    	fprintf(o1, "%4d ",dum);
        	    	fprintf(o1, "%4d %4d %4d %4d %4d\n",
			    dum3 + j + num_ver_fin + 1,
			    dum4 + j + num_ver_fin + 1,
			    dum4 + j + 1 + num_ver_fin + 1,
			    dum3 + j + 1 + num_ver_fin + 1,0);
		    	++dum;
		    }
		}
	    }
	}

	dum = 0;
	Y=0.0;
        fprintf(o1, "node no. coordinates\n");
        for( i = 0; i < num_hor+1; ++i )
        {
	    angle=angle_start;
            for( i2 = 0; i2 < num_fin; ++i2 )
            {
/* Draw fin for one section */

            	for( j = 0; j < num_ver_fin+1; ++j )
            	{
/* Rotate coordinates for left side and write coordinate */

			fdum2 = ((double)j)*length_inc_fin;
			X =(radius + fdum2)*cos(angle) - .5*thick_fin*sin(angle);
			Z =(radius + fdum2)*sin(angle) + .5*thick_fin*cos(angle);
       			fprintf(o1, "%4d ",dum);
			fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			++dum;
            	}

	    	angle -= angle_inc_fin;

            	for( j = 0; j < num_ver_fin+1; ++j )
            	{
/* Rotate fin coordinates for right side and write coordinate */

			fdum2 = ((double)j)*length_inc_fin;
			X =(radius + fdum2)*cos(angle) + .5*thick_fin*sin(angle);
			Z =(radius + fdum2)*sin(angle) - .5*thick_fin*cos(angle);

        		fprintf(o1, "%4d ",dum);
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
			++dum;
            	}

	    	angle -= angle_inc_cyl;
	    	fdum = angle;

            	for( j = 0; j < num_ver_div-1; ++j )
            	{
			X = radius*cos(angle);
			Z = radius*sin(angle);
        		fprintf(o1, "%4d ",dum);
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	   		angle -=angle_inc_cyl;
			++dum;
            	}

            	if( i2 == num_fin - 1 )
            	{
	    		X = radius*cos(angle);
	    		Z = radius*sin(angle);
            		fprintf(o1, "%4d ",dum);
            		fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	    		angle -=angle_inc_cyl;
	    		++dum;
		}

	    	angle = fdum;

            	for( j = 0; j < num_ver_div-1; ++j )
            	{
			X=(radius+thick)*cos(angle);
			Z=(radius+thick)*sin(angle);
        		fprintf(o1, "%4d ",dum);
                	fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	   		angle -=angle_inc_cyl;
			++dum;
            	}

            	if( i2 == num_fin - 1 )
            	{
	    	    X=(radius+thick)*cos(angle);
	    	    Z=(radius+thick)*sin(angle);
            	    fprintf(o1, "%4d ",dum);
            	    fprintf(o1, "%9.5f %9.5f %9.5f \n",X,Y,Z);
	    	    angle -=angle_inc_cyl;
	    	    ++dum;
            	}
            }

	    Y += length_inc;
        }


        fprintf( o1, "prescribed displacement x: node  disp value\n");
        for( i = 0; i < num_hor+1; ++i )
        {
		dum = i*numnp_layer;	
		fprintf( o1,"%5d %16.4e \n", dum, 0.0);
		fprintf( o1,"%5d %16.4e \n", dum+1, 0.0);
	}
        fprintf( o1, "%4d\n ",-10);

        fprintf( o1, "prescribed displacement y: node  disp value\n");
	fprintf( o1,"%5d %16.4e \n", 0, 0.0);
        fprintf( o1, "%4d\n ",-10);

        fprintf( o1, "prescribed displacement z: node  disp value\n");
        for( i = 0; i < num_hor+1; ++i )
        {
		dum = numnp_layer - 1 + i*numnp_layer;	
		dum2 = numnp_layer - 1 - num_ver_div + i*numnp_layer;	
		fprintf( o1,"%5d %16.4e \n", dum, 0.0);
		fprintf( o1,"%5d %16.4e \n", dum2, 0.0);
	}
        fprintf( o1, "%4d\n ",-10);

	if(el_type > 1 )
	{
            fprintf( o1, "prescribed temperature: node  temperature\n");
            fprintf( o1, "%4d\n ",-10);

            fprintf(o1, "surface node with bulk temperature: node  temperature\n");
	    dum = 0;
            for( i = 0; i < num_hor; ++i )
            {
                for( i2 = 0; i2 < num_fin; ++i2 )
                {
		    dum3 = numnp_layer*i + numnp_sec*i2;
            	    for( j = 1; j < num_ver_fin+1; ++j )
            	    {
        	    	fprintf(o1, "%4d   %12.6e\n", dum3 + j, TB);
		    }

		    dum3 = numnp_layer*i + numnp_sec*i2 + num_ver_fin + 1;
            	    for( j = 1; j < num_ver_fin+1; ++j )
            	    {
        	    	fprintf(o1, "%4d   %12.6e\n", dum3 + j, TB);
		    }
		}
	    }
            fprintf( o1, "%4d\n ",-10);
	}

        fprintf( o1, "node with point load and load vector in x,y,z\n");
        fprintf( o1, "%4d\n ",-10);

	if(el_type > 1 )
	{
            fprintf( o1, "node with heat load Q \n");
            fprintf( o1, " -10\n");

	    fprintf( o1, "node with heat generation \n");
            for( i = 0; i < num_hor; ++i )
            {
                for( i2 = 0; i2 < num_fin; ++i2 )
                {
		    dum3 = numnp_layer*i + numnp_sec*i2;
        	    fprintf(o1, "%4d   %12.6e\n", dum3, heat_node);
        	    fprintf(o1, "%4d   %12.6e\n", dum3 + num_ver_fin + 1, heat_node);
		    dum3 = numnp_layer*i + numnp_sec*i2 + 2*(num_ver_fin + 1);
            	    for( j = 0; j < num_ver_div - 1; ++j )
            	    {
        	    	fprintf(o1, "%4d   %12.6e\n", dum3 + j, heat_node);
		    }

		    if( i2 == num_fin - 1 )
		    {
		        dum3 = numnp_layer*i;
        	    	fprintf(o1, "%4d   %12.6e\n",
				dum3 + numnp_layer - num_ver_div - 1, heat_node);
		    }
		}
	    }
            fprintf( o1, " -10\n");

            fprintf( o1, "element with heat generation\n");
            fprintf( o1, " -10\n");
	}

        fprintf( o1, "element and gauss pt. with stress and stress vector xx,yy,zz,xy,zx,yz\n");
        fprintf( o1, "%4d ",-10);

        return 1;
}

main(int argc, char** argv)
{
	int check;
	check=writer();
} 

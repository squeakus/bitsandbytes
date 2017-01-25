/*
   This library function writes the resulting data for a finite element
   program which does analysis on a shell element 

		Updated 9/27/08

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

#define      OLD_COMPARE    0

extern int dof, integ_flag, doubly_curved_flag, nmat, nmode, numel, numnp;
extern int static_flag, element_stress_print_flag, gauss_stress_flag, flag_quad_element;

int shwriter ( BOUND bc, int *connect, double *coord, int *el_matl, double *force,
	int *id, double *fiber_xyz, MATL *matl, char *name, STRAIN *strain,
	SDIM *strain_node, STRESS *stress, SDIM *stress_node, double *U, double *Uz_fib)
{
	int i,j,dum,check, node, name_length;
	char *ccheck;
	double fpointx, fpointy, fpointz, fdum;
	char out[30], stress_dat[40], stress_type[6];
	FILE *o3, *o4;

/* Every output file is named after the input file with
   a ".osh" extension */

	name_length = strlen(name);
	if( name_length > 25) name_length = 25;

	memset(out,0,30*sizeof(char));

	ccheck = strncpy(out, name, name_length);
	if(!ccheck) printf( " Problems with strncpy \n");

	ccheck = strncpy(out+name_length, ".osh", 4);
	if(!ccheck) printf( " Problems with strncpy \n");

	o3 = fopen( out,"w" );

	fprintf( o3, "   numel numnp nmat nmode integ_flag");
	fprintf( o3, " (This is for the shell mesh file: %s)\n", name);
	fprintf( o3, "    %4d %4d %4d %4d %4d\n ", numel, numnp, nmat, nmode, integ_flag );
	fprintf( o3, " matl no., E mod., Poiss. Ratio, density, thickness, shear fac.\n");
	for( i = 0; i < nmat; ++i )
	{
	   fprintf( o3, "  %4d    %12.6e %12.6e %12.6e", i, matl[i].E,
		matl[i].nu, matl[i].rho);
	   if( matl[i].extrathick > 0 )
	   {
		fprintf( o3, " %12.6e %12.6e\n", matl[i].thick, matl[i].shear);
	   }
	   else
	   {
		fprintf( o3, " %12.6e\n", matl[i].shear);
	   }
	}

	fprintf( o3, "el no., connectivity, matl no. \n");

	if( !flag_quad_element )
	{
/* Triangle elements */
	    for( i = 0; i < numel; ++i )
	    {
	       fprintf( o3, "%6d ",i);
	       for( j = 0; j < npell3; ++j )
	       {
		    fprintf( o3, "%6d ",*(connect+npell3*i+j));
	       }
	       fprintf( o3, "   %3d\n",*(el_matl+i));
	    }
	}
	else
	{
	    for( i = 0; i < numel; ++i )
	    {
	       fprintf( o3, "%6d ",i);
	       for( j = 0; j < npell4; ++j )
	       {
		    fprintf( o3, "%6d ",*(connect+npell4*i+j));
	       }
	       fprintf( o3, "   %3d\n",*(el_matl+i));
	    }
	}

	fprintf( o3, "node no., coordinates \n");
	for( i = 0; i < numnp; ++i )
	{
/* I have tried 4 methods of calculating the updated coordinates. */

#if 0

/* The first is simply adding the nodal U which comes from the solution vecter.
*/
	   fpointx = *(coord+nsd*i) + *(U+ndof*i);
	   fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1);
	   fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
	   fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		i, fpointx, fpointy, fpointz );

/* The second adds to the x and y coordinates the global displacement in z multiplied
   by the local angle y and x respectively.
*/
	   fpointx = *(coord+nsd*i) + *(U+ndof*i) +
		*(U+ndof*i+2)*(*(U+ndof*i+4));
	   fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1) -
		*(U+ndof*i+2)*(*(U+ndof*i+3));
	   fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
	   fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		i, fpointx, fpointy, fpointz );
#endif

#if OLD_COMPARE
/* The third adds to the x and y coordinates the normal projection of the displacement
   vector in the fiber direction multiplied by the local angle y and x respectively.
*/
	   fpointx = *(coord+nsd*i) + *(U+ndof*i) +
		*(Uz_fib+i)*(*(U+ndof*i+4));
	   fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1) -
		*(Uz_fib+i)*(*(U+ndof*i+3));
	   fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2);
	   fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		i, fpointx, fpointy, fpointz );

#endif

#if !OLD_COMPARE
/* The fourth method is probably the most correct.  It is based on equations 6.2.20-6.2.24 of
   "The Finite Element Method" by Thomas Hughes, page 388-389.  (David Benson, in "AMES 232 B
   Course Notes" page 50, does the same as Hughes.)  I have made some modifications,
   specifically, I replace za in equation 6.2.23 with "Uz_fib" which is the projection of the
   displacement at a node in the fiber "e3" direction. 

   It should be noted that in the "ANSYS User's Manual", page 12-12, these same equations mistakenly
   switch the fiber vectors e1 and e2(which are called "a" and "b" in ANSYS).  The first time these
   equations appear are in 12.5.1, 12.5.2, 12.5.3.  They also appear incorrectly like this later.
*/
	   fdum = - *(fiber_xyz + nsdsq*i + 1*nsd)*(*(U+ndof*i+3)) +
		*(fiber_xyz + nsdsq*i)*(*(U+ndof*i+4)); 
	   fpointx = *(coord+nsd*i) + *(U+ndof*i) +
		*(Uz_fib+i)*fdum;
	   fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 1)*(*(U+ndof*i+3)) +
		*(fiber_xyz + nsdsq*i + 1)*(*(U+ndof*i+4)); 
	   fpointy = *(coord+nsd*i+1) + *(U+ndof*i+1) +
		*(Uz_fib+i)*fdum;
	   fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 2)*(*(U+ndof*i+3)) +
		*(fiber_xyz + nsdsq*i + 2)*(*(U+ndof*i+4)); 
	   fpointz = *(coord+nsd*i+2) + *(U+ndof*i+2) +
		*(Uz_fib+i)*fdum;
	   fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		i, fpointx, fpointy, fpointz );
#endif

	}

	if(doubly_curved_flag)
	{
	    fprintf( o3, "The corresponding top nodes are: \n");
	    for( i = 0; i < numnp; ++i )
	    {
/* I have tried 4 methods of calculating the updated coordinates. */

#if 0

/* The first is simply adding the nodal U which comes from the solution vecter.
*/
		fpointx = *(coord+nsd*(numnp+i)) + *(U+ndof*i);
		fpointy = *(coord+nsd*(numnp+i)+1) + *(U+ndof*i+1);
		fpointz = *(coord+nsd*(numnp+i)+2) + *(U+ndof*i+2);
		fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		    i, fpointx, fpointy, fpointz );

/* The second adds to the x and y coordinates the global displacement in z multiplied
   by the local angle y and x respectively.
*/
		fpointx = *(coord+nsd*(numnp+i)) + *(U+ndof*i) +
		    *(U+ndof*i+2)*(*(U+ndof*i+4));
		fpointy = *(coord+nsd*(numnp+i)+1) + *(U+ndof*i+1) -
		    *(U+ndof*i+2)*(*(U+ndof*i+3));
		fpointz = *(coord+nsd*(numnp+i)+2) + *(U+ndof*i+2);
		fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		    i, fpointx, fpointy, fpointz );
#endif

#if OLD_COMPARE
/* The third adds to the x and y coordinates the normal projection of the displacement
   vector in the fiber direction multiplied by the local angle y and x respectively.
*/
		fpointx = *(coord+nsd*(numnp+i)) + *(U+ndof*i) +
		    *(Uz_fib+i)*(*(U+ndof*i+4));
		fpointy = *(coord+nsd*(numnp+i)+1) + *(U+ndof*i+1) -
		    *(Uz_fib+i)*(*(U+ndof*i+3));
		fpointz = *(coord+nsd*(numnp+i)+2) + *(U+ndof*i+2);
		fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		    i, fpointx, fpointy, fpointz );

#endif

#if !OLD_COMPARE

/* See notes above for the top nodes. */

		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i)*(*(U+ndof*i+4));
		fpointx = *(coord+nsd*(numnp+i)) + *(U+ndof*i) +
		    *(Uz_fib+i)*fdum;
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 1)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i + 1)*(*(U+ndof*i+4));
		fpointy = *(coord+nsd*(numnp+i)+1) + *(U+ndof*i+1) +
		    *(Uz_fib+i)*fdum;
		fdum = - *(fiber_xyz + nsdsq*i + 1*nsd + 2)*(*(U+ndof*i+3)) +
		    *(fiber_xyz + nsdsq*i + 2)*(*(U+ndof*i+4));
		fpointz = *(coord+nsd*(numnp+i)+2) + *(U+ndof*i+2) +
		    *(Uz_fib+i)*fdum;
		fprintf( o3, "%4d %14.6f %14.6f %14.6f\n",
		    i, fpointx, fpointy, fpointz );
#endif
	    }
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

	fprintf( o3, "prescribed angle phi x: node angle value\n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+3));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "prescribed angle phi y: node angle value \n");
	for( i = 0; i < numnp; ++i )
	{
		fprintf( o3, "%4d %14.6e\n", i,
			*(U+ndof*i+4));
	}
	fprintf( o3, " -10 \n");

	fprintf( o3, "node with point load x, y, z and 2 moments phi x, phi y \n");

	if( static_flag)
	{
	    for( i = 0; i < bc.num_force[0] ; ++i )
	    {
		node = bc.force[i];
		*(id+ndof*node) = -1;
		*(id+ndof*node+1) = -1;
		*(id+ndof*node+2) = -1;
		*(id+ndof*node+3) = -1;
		*(id+ndof*node+4) = -1; 
	    }
	}

	for( i = 0; i < numnp; ++i )
	{
	   if( *(id+ndof*i) < 0 || *(id+ndof*i+1) < 0 || *(id+ndof*i+2) < 0  ||
		*(id+ndof*i+3) < 0 || *(id+ndof*i+4) < 0  )
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

	fprintf( o3, "node no. with stress ");
	fprintf( o3, "and stress vector in lamina xx,yy,xy,zx,yz \n");
	for( i = 0; i < 2*numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",stress_node[i].xx);
		fprintf( o3,"%14.6e ",stress_node[i].yy);
		fprintf( o3,"%14.6e ",stress_node[i].xy);
		fprintf( o3,"%14.6e ",stress_node[i].zx);
		fprintf( o3,"%14.6e ",stress_node[i].yz);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with stress ");
	fprintf( o3, "and principal stress I,II,III \n");
	for( i = 0; i < 2*numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",stress_node[i].I);
		fprintf( o3,"%14.6e ",stress_node[i].II);
		fprintf( o3,"%14.6e ",stress_node[i].III);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with strain ");
	fprintf( o3, "and strain vector in lamina xx,yy,xy,zx,yz \n");
	for( i = 0; i < 2*numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",strain_node[i].xx);
		fprintf( o3,"%14.6e ",strain_node[i].yy);
		fprintf( o3,"%14.6e ",strain_node[i].xy);
		fprintf( o3,"%14.6e ",strain_node[i].zx);
		fprintf( o3,"%14.6e ",strain_node[i].yz);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");
	fprintf( o3, "node no. with strain ");
	fprintf( o3, "and principal strain I,II,III \n");
	for( i = 0; i < 2*numnp; ++i )
	{
		fprintf( o3,"%4d  ",i);
		fprintf( o3,"%14.6e ",strain_node[i].I);
		fprintf( o3,"%14.6e ",strain_node[i].II);
		fprintf( o3,"%14.6e ",strain_node[i].III);
		fprintf( o3, "\n");
	}
	fprintf( o3, " -10 \n");

	if(element_stress_print_flag)
	{

		memset(stress_type, 0, 6*sizeof(char));
		if(gauss_stress_flag)
		{
		    ccheck = strncpy(stress_type, "gauss", 5);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}
		else
		{
		    ccheck = strncpy(stress_type, "nodal", 5);
		    if(!ccheck) printf( " Problems with strncpy \n");
		}

/* Open stress data output file */

		memset(stress_dat,0,40*sizeof(char));

		ccheck = strncpy(stress_dat, name, name_length);
		if(!ccheck) printf( " Problems with strncpy \n");

		ccheck = strncpy(stress_dat+name_length, ".str.osh", 8);
		if(!ccheck) printf( " Problems with strncpy \n");

		o4 = fopen( stress_dat,"w" );

		fprintf( o4, "element no. and %5s pt. no. with stress ", stress_type);
		fprintf( o4, "and stress vector in lamina xx,yy,xy,zx,yz \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int8; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",stress[i].pt[j].xx);
			fprintf( o4,"%14.6e ",stress[i].pt[j].yy);
			fprintf( o4,"%14.6e ",stress[i].pt[j].xy);
			fprintf( o4,"%14.6e ",stress[i].pt[j].zx);
			fprintf( o4,"%14.6e ",stress[i].pt[j].yz);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with stress ", stress_type);
		fprintf( o4, "and principal stress I,II,III \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int8; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",stress[i].pt[j].I);
			fprintf( o4,"%14.6e ",stress[i].pt[j].II);
			fprintf( o4,"%14.6e ",stress[i].pt[j].III);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with strain ", stress_type);
		fprintf( o4, "and strain vector in lamina xx,yy,xy,zx,yz \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int8; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",strain[i].pt[j].xx);
			fprintf( o4,"%14.6e ",strain[i].pt[j].yy);
			fprintf( o4,"%14.6e ",strain[i].pt[j].xy);
			fprintf( o4,"%14.6e ",strain[i].pt[j].zx);
			fprintf( o4,"%14.6e ",strain[i].pt[j].yz);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
		fprintf( o4, "element no. and %5s pt. no. with strain ", stress_type);
		fprintf( o4, "and principal strain I,II,III \n");
		for( i = 0; i < numel; ++i )
		{
		   for( j = 0; j < num_int8; ++j )
		   {
			fprintf( o4,"%4d %4d  ",i,j);
			fprintf( o4,"%14.6e ",strain[i].pt[j].I);
			fprintf( o4,"%14.6e ",strain[i].pt[j].II);
			fprintf( o4,"%14.6e ",strain[i].pt[j].III);
			fprintf( o4, "\n");
		   }
		}
		fprintf( o4, " -10 \n");
	}

	return 1;
}


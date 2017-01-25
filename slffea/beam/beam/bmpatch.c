/*
    This library function creates data for patch test analysis on a
    beam element .  First, it generates the coordinates, then
    it creates the prescribed displacements so that the main
    finite element program can do the analysis.  It also splits
    the patch test into 2, one for x,y,z coordintates and one for
    phi x,y,z angles.

		Updated 9/10/04

    SLFFEA source file
    Version:  1.5
    Copyright (C) 1999, 2000, 2001, 2002  San Le 

    The source code contained in this file is released under the
    terms of the GNU Library General Public License.
 
*/

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "bmconst.h"
#include "bmstruct.h"


int main(int argc, char** argv)
{
	int dof, nmat, nmode, numel, numnp, integ_flag;
	int i,j,dum;
	double fdum1, fdum2, fdum3, ux[9], uy[9], uz[9],
		uphix[9], uphiy[9], uphiz[9];
	double x[3], y[3], z[3], L, m, m2, b, b2;
	char name[30];
	char buf[ BUFSIZ ];
	FILE *o2, *o3, *o4;
	char text;

	numel = 2;
	numnp = 3;
	nmat = 1;
	nmode = 0;

	m = -0.5017;
	b = 3.292;
	m2 = 1.78392;
	b2 = -2.283;

	*(x) = -4.282;
	*(x+1) = 8.0923;
	*(x+2) = 14.759;

	o2 = fopen( "patch","w" );
	o3 = fopen( "patch.xyz","w" );
	o4 = fopen( "patch.phi","w" );

	fprintf( o2, "      numel numnp nmat nmode This is the patch test \n ");
	fprintf( o3, "      numel numnp nmat nmode This is the patch test \n ");
	fprintf( o4, "      numel numnp nmat This is the patch test \n ");
	fprintf( o2, "    %4d %4d %4d %4d\n ",numel,numnp,nmat,nmode);
	fprintf( o3, "    %4d %4d %4d %4d\n ",numel,numnp,nmat,nmode);
	fprintf( o4, "    %4d %4d %4d %4d\n ",numel,numnp,nmat,nmode);

	fprintf( o2, " matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n");
	fprintf( o3, " matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n");
	fprintf( o4, " matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n");
	fprintf( o2, "%3d ",0);
	fprintf( o3, "%3d ",0);
	fprintf( o4, "%3d ",0);
	fprintf( o2, " %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",200000.0,.27,3.44e3,.01,.001,.001);
	fprintf( o3, " %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",200000.0,.27,3.44e3,.01,.001,.001);
	fprintf( o4, " %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",200000.0,.27,3.44e3,.01,.001,.001);

	fprintf( o2, "el no., connectivity, matl no. \n");
	fprintf( o3, "el no., connectivity, matl no. \n");
	fprintf( o4, "el no., connectivity, matl no. \n");
	fprintf( o2, "%4d %4d %4d %4d %4d\n",0,0,1,0,2);
	fprintf( o2, "%4d %4d %4d %4d %4d\n",1,1,2,0,2);
	fprintf( o3, "%4d %4d %4d %4d %4d\n",0,0,1,0,2);
	fprintf( o3, "%4d %4d %4d %4d %4d\n",1,1,2,0,2);
	fprintf( o4, "%4d %4d %4d %4d %4d\n",0,0,1,0,2);
	fprintf( o4, "%4d %4d %4d %4d %4d\n",1,1,2,0,2);
	fprintf( o2, "node no., coordinates \n");
	fprintf( o3, "node no., coordinates \n");
	fprintf( o4, "node no., coordinates \n");
	for( i = 0; i < numnp; ++i )
	{
	   fprintf( o2, "%d ",i);
	   fprintf( o3, "%d ",i);
	   fprintf( o4, "%d ",i);
	   *(y+i) = m*(*(x+i)) + b;
	   L = (*(x+i)-*(x))*(*(x+i)-*(x)) + (*(y+i)-*(y))*(*(y+i)-*(y));
	   L = sqrt(L);
	   *(z+i) = m2*L + b2;
	   fdum1 = *(x+i);
	   fdum2 = *(y+i);
	   fdum3 = *(z+i);
	   
	   fprintf( o2, "%14.6f %14.6f %14.6f",fdum1, fdum2, fdum3);
	   fprintf( o3, "%14.6f %14.6f %14.6f",fdum1, fdum2, fdum3);
	   fprintf( o4, "%14.6f %14.6f %14.6f",fdum1, fdum2, fdum3);
	   *(ux+i) = 0.003*fdum1 - 0.040*fdum2 + 0.021*fdum3;
	   *(uy+i) = -0.011*fdum1 - 0.020*fdum2 + 0.035*fdum3;
	   *(uz+i) = 0.041*fdum1 + 0.033*fdum2 - 0.027*fdum3;
	   *(uphix+i) = 1.000e-04*fdum1 + 3.000e-04*fdum2 - 1.800e-04*fdum3;
	   *(uphiy+i) = 2.000e-04*fdum1 + 4.000e-04*fdum2 + 3.300e-04*fdum3;
	   *(uphiz+i) = 7.100e-04*fdum1 - 6.500e-04*fdum2 + 2.700e-04*fdum3;
	   fprintf( o2, "\n");
	   fprintf( o3, "\n");
	   fprintf( o4, "\n");
	}

	fprintf( o2, "element with specified local z axis: x, y, z component\n");
	fprintf( o3, "element with specified local z axis: x, y, z component\n");
	fprintf( o4, "element with specified local z axis: x, y, z component\n");
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);

	fprintf( o2, "prescribed displacement x: node  disp value\n");
	fprintf( o3, "prescribed displacement x: node  disp value\n");
	fprintf( o4, "prescribed displacement x: node  disp value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(ux+i));
		fprintf( o3, "%4d %14.6e\n",i,*(ux+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(ux+i));
		fprintf( o3, "%4d %14.6e\n",i,*(ux+i));
	}
	fprintf( o4, "%4d %14.6e\n",0,0.0);
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "prescribed displacement y: node  disp value\n");
	fprintf( o3, "prescribed displacement y: node  disp value\n");
	fprintf( o4, "prescribed displacement y: node  disp value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uy+i));
		fprintf( o3, "%4d %14.6e\n",i,*(uy+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uy+i));
		fprintf( o3, "%4d %14.6e\n",i,*(uy+i));
	}
	fprintf( o4, "%4d %14.6e\n",0,0.0);
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "prescribed displacement z: node  disp value\n");
	fprintf( o3, "prescribed displacement z: node  disp value\n");
	fprintf( o4, "prescribed displacement z: node  disp value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uz+i));
		fprintf( o3, "%4d %14.6e\n",i,*(uz+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uz+i));
		fprintf( o3, "%4d %14.6e\n",i,*(uz+i));
	}
	fprintf( o4, "%4d %14.6e\n",0,0.0);
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "prescribed angle phi x: node angle value\n");
	fprintf( o3, "prescribed angle phi x: node angle value\n");
	fprintf( o4, "prescribed angle phi x: node angle value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphix+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphix+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphix+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphix+i));
	}
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "prescribed angle phi y: node angle value\n");
	fprintf( o3, "prescribed angle phi y: node angle value\n");
	fprintf( o4, "prescribed angle phi y: node angle value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphiy+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphiy+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphiy+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphiy+i));
	}
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "prescribed angle phi z: node angle value\n");
	fprintf( o3, "prescribed angle phi z: node angle value\n");
	fprintf( o4, "prescribed angle phi z: node angle value\n");
	for( i = 0; i < 1; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphiz+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphiz+i));
	}
	for( i = 2; i < numnp; ++i )
	{
		fprintf( o2, "%4d %14.6e\n",i,*(uphiz+i));
		fprintf( o4, "%4d %14.6e\n",i,*(uphiz+i));
	}
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);


	fprintf( o2, "node with point load x, y, z and 3 moments phi x, phi y, phi z\n");
	fprintf( o3, "node with point load x, y, z and 3 moments phi x, phi y, phi z\n");
	fprintf( o4, "node with point load x, y, z and 3 moments phi x, phi y, phi z\n");
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "element with distributed load in local beam y and z coordinates\n");
	fprintf( o3, "element with distributed load in local beam y and z coordinates\n");
	fprintf( o4, "element with distributed load in local beam y and z coordinates\n");
	fprintf( o2, "%4d\n ",-10);
	fprintf( o3, "%4d\n ",-10);
	fprintf( o4, "%4d\n ",-10);
	fprintf( o2, "element no. and gauss pt. no. with local stress xx and ");
	fprintf( o3, "element no. and gauss pt. no. with local stress xx and ");
	fprintf( o4, "element no. and gauss pt. no. with local stress xx and ");
	fprintf( o2, "moment xx,yy,zz\n");
	fprintf( o3, "moment xx,yy,zz\n");
	fprintf( o4, "moment xx,yy,zz\n");
	fprintf( o2, "%4d ",-10);
	fprintf( o3, "%4d ",-10);
	fprintf( o4, "%4d ",-10);

	fprintf( o2, "\n\n\n Note: Remove the prescribed angles and disp. for node 1.\n");
	fprintf( o2, "%4d %14.6e\n",1,*(ux+1));
	fprintf( o2, "%4d %14.6e\n",1,*(uy+1));
	fprintf( o2, "%4d %14.6e\n",1,*(uz+1));
	fprintf( o2, "%4d %14.6e\n",1,*(uphix+1));
	fprintf( o2, "%4d %14.6e\n",1,*(uphiy+1));
	fprintf( o2, "%4d %14.6e\n",1,*(uphiz+1));
	fprintf( o3, "\n\n\n Note: Remove the prescribed angles and disp. for node 1.\n");
	fprintf( o3, "%4d %14.6e\n",1,*(ux+1));
	fprintf( o3, "%4d %14.6e\n",1,*(uy+1));
	fprintf( o3, "%4d %14.6e\n",1,*(uz+1));
	fprintf( o4, "\n\n\n Note: Remove the prescribed angles and disp. for node 1.\n");
	fprintf( o4, "%4d %14.6e\n",1,*(uphix+1));
	fprintf( o4, "%4d %14.6e\n",1,*(uphiy+1));
	fprintf( o4, "%4d %14.6e\n",1,*(uphiz+1));
	fprintf( o2, "\n You also need to divide this file into 2 patch test");
	fprintf( o2, "\n The first you can call patch.xyz and remove all the");
	fprintf( o2, "\n prescribed angles phi.  The second, call patch.phi");
	fprintf( o2, "\n and remove the prescribed x, y, z.");
	fprintf( o2, "\n");

        return 1;
}


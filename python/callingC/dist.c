#include <stdio.h>
#include <math.h>

//Function Declarations

int distance(int,int,int,int,int,int);

//Distance Function

int distance(int x1, int y1, int z1, int x2, int y2, int z2)
{	
	int diffx = x1 - x2;
        int diffy = y1 - y2;
	int diffz = z1 - z2;
	int diffx_sqr = pow(diffx,2);
	int diffy_sqr = pow(diffy,2);
	int diffz_sqr = pow(diffz,2);
	int distance = sqrt(diffx_sqr + diffy_sqr + diffz_sqr);

return distance;
}

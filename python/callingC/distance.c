#include <stdio.h>
#include <math.h>

//Function Declarations

double CalculateDistance(double, double, double, double);

int euclideanDistance(int,int,int,int,int,int);

int main ()
{
/* Function Call */
/*double distance1 = CalculateDistance( 0.0, 0.0, 5.0, 5.0);
double distance2 = CalculateDistance(0.0, 0.0, 5.0, 0.0);
double distance3 = CalculateDistance(0.0, 0.0, 0.0, 5.0);*/

  int distance1 = euclideanDistance(1000,0,0,0,0,0);
  int distance2 = euclideanDistance(0,1000,0,0,0,0);
  int distance3 = euclideanDistance(1000,1000,1000,0,0,0);


printf("      Point1	      Point 2		Distance\n");
 printf("(%d, %d %d)",1000, 0, 0);
printf("(%d, %d %d)", 0, 0, 0);
printf("%d\n", distance1); 

 printf("(%d, %d %d)",0,1000,0);
printf("(%d, %d %d)", 0, 0, 0);
printf("%d\n", distance2);

 printf("(%d, %d %d)",1000,1000,1000);
printf("(%d, %d %d)", 0, 0, 0);
printf("%d\n", distance3);	
	
return 0;
}

//Distance Function

double CalculateDistance(double x1, double y1, double x2, double y2)
{	
	double diffx = x1 - x2;
	double diffy = y1 - y2;
	double diffx_sqr = pow(diffx,2);
	double diffy_sqr = pow(diffy,2);
	double distance = sqrt(diffx_sqr + diffy_sqr);

return distance;
}

int euclideanDistance(int x1, int y1, int z1, int x2, int y2, int z2)
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

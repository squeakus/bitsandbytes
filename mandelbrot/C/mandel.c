/* This is a very simple program to create the mandelbrot set */

#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>

#define width 640
#define height 480

main()
{
  double x,y;
  double xstart,xstep,ystart,ystep;
  double xend, yend;
  double z,zi,newz,newzi;
  double colour;
  int iter;
  long col;
  char pic[height][width][3];
  int i,j,k;
  int inset;
  int fd;
  char buffer[100];

  /* Read in the initial data */
  printf("Enter xstart, xend, ystart, yend, iterations: ");
  if (scanf("%lf%lf%lf%lf%d", &xstart, &xend, &ystart, &yend, &iter) != 5)
  {
    printf("Error!\n");
    exit(1);
  }

  /* these are used for calculating the points corresponding to the pixels */
  xstep = (xend-xstart)/width;
  ystep = (yend-ystart)/height;

  /*the main loop */
  x = xstart;
  y = ystart;
  for (i=0; i<height; i++)
  {
    printf("Now on line: %d\n", i);
    for (j=0; j<width; j++)
    {
      z = 0;
      zi = 0;
      inset = 1;
      for (k=0; k<iter; k++)
      {
        /* z^2 = (a+bi)(a+bi) = a^2 + 2abi - b^2 */
	newz = (z*z)-(zi*zi) + x;
	newzi = 2*z*zi + y;
        z = newz;
        zi = newzi;
	if(((z*z)+(zi*zi)) > 4)
	{
	  inset = 0;
	  colour = k;
	  k = iter;
	}
      }
      if (inset)
      {
	pic[i][j][0] = 0;
	pic[i][j][1] = 0;
	pic[i][j][2] = 0;
      }
      else
      { 
	pic[i][j][0] = colour / iter * 255;
	pic[i][j][1] = colour / iter * 255 / 2;
	pic[i][j][2] = colour / iter * 255 / 2;
      }
      x += xstep;
    }
    y += ystep;
    x = xstart;
  }

  /* writes the data to a TGA file */
  if ((fd = open("mand.tga", O_RDWR+O_CREAT, 0)) == -1)
  {
    printf("error opening file\n");
    exit(1);
  }
  buffer[0] = 0;
  buffer[1] = 0;
  buffer[2] = 2;
  buffer[8] = 0; buffer[9] = 0;
  buffer[10] = 0; buffer[11] = 0;
  buffer[12] = (width & 0x00FF); buffer[13] = (width & 0xFF00) >> 8;
  buffer[14] = (height & 0x00FF); buffer[15] = (height & 0xFF00) >> 8;
  buffer[16] = 24;
  buffer[17] = 0;
  write(fd, buffer, 18);
  write(fd, pic, width*height*3);
  close(fd);
} 

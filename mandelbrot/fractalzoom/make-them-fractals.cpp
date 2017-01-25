// (C) 2008 Sean Brennan
// This is a home brew mandelbrot set renderer.  Considering the vastness
// of better renderers out there, this is purely functional code and is quite
// ugly.  Usage is:
// ./make-them-fractals lower_left.re lower_left.im upper_right.re upper_right.im xpixels ypixels anti_alias_len
// Fractals are noisy and genrally unruly so facilities to blur them are
// controlled by the anti_alias_len parameter, which loops a blur.
// Driven by make-fractals.py

#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;

#define MAXXX 8;
#define EPSILON 0.0000000001

class complex {
  public:
  	double re, im;
  complex(void) { re = 0.0, im = 0.0;}
  complex(double f) { re = f, im = 0.0;};
  complex(double r, double i) { re = r, im = i;};
  complex(const complex &c) { re = c.re; im = c.im;};

  friend complex operator*(complex a, complex b);
  friend complex operator+(complex a, complex b);
  friend complex operator/(complex a, complex b);
  friend complex operator-(complex a, complex b);
};

complex operator-(complex a, complex b)
	{ return complex(a.re - b.re, a.im - b.im); }

complex operator+(complex a, complex b)
	{ return complex(a.re + b.re, a.im + b.im); }

complex operator-(complex a) { return complex( - a); }

complex operator*(complex a, complex b) {
  return complex(a.re * b.re - a.im * b.im , a.re * b.im + a.im * b.re);
}

complex operator/(complex a, complex b) {
  /* derivation of a/b where a,b are complex:
  a = d + ei, b = x + zi
  a/b = (d + ei) / (x + zi) = d / (x + zi) + ei / (x + zi) 
      = d(x-zi)/((x+zi)(x-zi)) + ei(x-zi)/((x+zi)(x-zi)) multiply top and bot
      = d(x-zi)/(x^2+z^2) + ei(x-zi)/(x^2+z^2)  convert denom to real number
      = (dx+ez)/(x^2+z^2) + (ex-dz)/(x^2+z^2) regroup into real/imaginary
  */
  float xxzz = b.re * b.re + b.im * b.im;
  if (xxzz < EPSILON) 
  	return complex(1.0, 0.0);
  xxzz = 1.0 / xxzz;
  return complex((a.re * b.re + a.im * b.im) * xxzz,
	(a.im * b.re - a.re * b.im)  * xxzz);
}

// define corners of area of interest
complex lower_left, upper_right;

//mandelbrot set is within this squared radius
double mandel_radius_2 = 4.0;
complex stepper(0.0, 0.0);
int maxiters = 2270; //ATTENTION: MAJOR PERF HERE< MInImIZE@@@@!!!!

int mandel_iter(complex c) {
  int count = 0;
  double radius;
  complex z(0.0, 0.0);
  do {
    z = z * z + c;
    radius = (z * z).re;
    count++;
    if (count > maxiters)
      break;
  } while (radius < mandel_radius_2);
  return count;
}

void blur(int *buf, int xres, int yres) {
   int x, y, xneg, yneg, xpos, ypos;
   int *newbuf = new int[xres * yres];
   int *oldbuf;

   //for (x = 0 ; x < xres * yres ; x++)
     //newbuf[x] = 0;

   for (y = 0 ; y < yres ; y++)
     for (x = 0 ; x < xres ; x++) {
       xneg = x - 1;
       yneg = y - 1;
       xpos = x + 1;
       ypos = y + 1;
       if (xneg < 0) xneg = 0;
       if (yneg < 0) yneg = 0;
       if (xpos == xres) xpos = x;
       if (ypos == yres) ypos = y;
       newbuf[x + xres * y] = (
		buf[xneg + xres * yneg] +
		buf[xneg + xres * y] +
		buf[xneg + xres * ypos] +
		buf[x + xres * yneg] +
		buf[x + xres * y] +
		buf[x + xres * ypos] +
		buf[xpos + xres * yneg] +
		buf[xpos + xres * y] +
		buf[xpos + xres * ypos] ) / 9;
     }
   for (x = 0 ; x < xres * yres ; x++)
     buf[x] = newbuf[x];
   delete [] newbuf;
}
 
       

int julia_iter(complex z, complex c) {
  int count = 0;
  double radius;
  do {
    z = z * z + c;
    radius = (z * z).re;
    count++;
    if (count > maxiters)
      break;
  } while (radius < mandel_radius_2);
  return count;
}

int anti_alias(int (*func) (complex), complex index, complex stepper, int dim) {
   complex q;
   int xdelta, ydelta, total;
   double xstep, ystep;

   q = index + stepper * -0.5;
   xstep = stepper.re * 0.5 / double(dim);
   ystep = stepper.im * 0.5 / double(dim);
   total = 0;
   for (ydelta = 0; ydelta < dim ; ydelta++)
     for (xdelta = 0; xdelta < dim ; xdelta++)
       total += func(q +  complex( (double)(( 2 * xdelta + 1) * xstep),
                                    (double)(( 2 * ydelta + 1) * ystep)));
        
   total /= dim * dim;
   return total;
} 


int main(int argc, char *argv[])
{

  if (argc != 9) {
    cout << "Usage: make-fractal -re -im +re +im xres yres alias outfile.pgm";
    cout << '\n';
    exit(1);
  }


  //ofstream outfile ("mb.pgm");
  ofstream outfile (argv[8]);
  
   
  //lower_left.re =  -1.50;
  //lower_left.im = -1.0;
  //upper_right.re = 0.5;
  //upper_right.im = 1.0;
  complex lower_left(0.0, 0.0);
  complex upper_right(0.0, 0.0);
  lower_left.re = atof(argv[1]);
  lower_left.im = atof(argv[2]);
  upper_right.re = atof(argv[3]);
  upper_right.im = atof(argv[4]);

  //lower_left = complex( -0.24567, 0.64196);
  //upper_right = complex(-0.22679, 0.66103);

  //lower_left = complex( -1.0, -1.0);
  //upper_right = complex(1.0, 1.0);

  complex julie(0.45, -0.1428);

  int miniters = maxiters;
  int count = 0;
  int debug = 0;

  complex index;
  complex z;
  int textmode = 0;
  int xpixels;
  int ypixels;
  int anti_alias_len;
  xpixels = atoi(argv[5]);
  ypixels = atoi(argv[6]);
  anti_alias_len = atoi(argv[7]);

  int bufsize;

  bufsize = xpixels * ypixels;
  cout << "bufsize: " << bufsize << '\n';

  int *pixel_buffer = new int[bufsize + 1]; //slop

  if (textmode) xpixels = 70;
  if (textmode) ypixels = 30;
  stepper.re = (upper_right.re - lower_left.re) / (double) xpixels;
  stepper.im = (upper_right.im - lower_left.im) / (double) ypixels;
  cout << "Making fractal " << argv[7] << " At [" << lower_left.re;
  cout << ", " << lower_left.im << "] ";
  cout << " by [" << upper_right.re;
  cout << ", " << upper_right.im << "] ";
  cout << " With res [" << xpixels;
  cout << ", " << ypixels << "] " << '\n';
  cout << " anti aliasing factor " << anti_alias_len << '\n';

  unsigned char c;
  int count_filtered;
  int count_red, count_green, count_blue; // ((blue * 256) + red) * 256 + green
  outfile << "P6\n" << xpixels << '\n' << ypixels << '\n' << "255\n";
  int im_count, re_count;
  index = lower_left;
  int ptmp, percent_done = 0;
  for  (im_count = 0 ; im_count < ypixels ; index.im += stepper.im, im_count++)
  {
  	for  (re_count = 0, index.re = lower_left.re ; re_count < xpixels ;
		index.re += stepper.re, re_count++)
	{
			count = anti_alias(&mandel_iter, index, stepper, anti_alias_len);
			//count = julia_iter(index, julie);
			//count_filtered = count;
                        pixel_buffer[re_count + im_count * xpixels] = count;
                        if (count < miniters) miniters = count;
	}
	ptmp = (int) ( im_count * 100.0 / ypixels);
	if (ptmp > percent_done)
	{
		cout << "done: " << ptmp << '\n';
		percent_done = ptmp;
	}
  }
  cout << "Min iter: " << miniters << '\n';
  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);

  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);
  blur(pixel_buffer,  xpixels, ypixels);
  for (im_count = 0 ; im_count < (xpixels * ypixels) ; im_count++) {
    count = pixel_buffer[im_count];
    count_blue = (count >> 16) & 0xff;
    count_red = (count >> 8) & 0xff;
    count_green = (count) & 0xff;
                        if (debug == 1) {
				cout << "P: " << index.re << " ";
				cout << index.im << " -> " << count << " -  "; 
				cout << count_red << " ";
				cout << count_green << " ";
				cout << count_blue << " " << '\n';
			}
    if (count < maxiters) {
  	   if (textmode) {
	      c = 'X';
	      outfile.put(c);
           } else {
              outfile.put((unsigned char) (count_red));
              outfile.put((unsigned char) (count_green));
              outfile.put((unsigned char) (count_blue));
	  }

   } else {
          c = 0xff;
          if (textmode) {
                   c = '-';
                   outfile.put(c);
          } else {
                   outfile.put(c);
                   outfile.put(c);
                   outfile.put(c);
          }
   }
  }
  if ((textmode) && ((im_count % xpixels) == 0)) outfile << '\n';

}

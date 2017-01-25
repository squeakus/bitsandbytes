#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL.h>

#include "vgalib.h"

// Number of iterations per pixel
// If you change this, change it in the ASM too...

#define ITERA 250

// Use C inner loop or SSE ?

// #define C_INNER

int MAXX;
int MAXY;

extern Uint8 *buffer;
Uint8 *previewBuffer = NULL;

//#define CENTERX  -.1704488254
//#define CENTERY  -1.074076051888

//#define CENTERX  -.151223346157
//#define CENTERY  -1.043256318884

//#define CENTERX  -.167269950706
//#define CENTERY  -1.041407957762

#define CENTERX  -1.758628987507
#define CENTERY  -0.019255356969

// SSE requires data to be aligned to 16bytes, so...

#ifdef __GNUC__
    #define DECLARE_ALIGNED(n,t,v)       t v __attribute__ ((aligned (n)))
#else
    #define DECLARE_ALIGNED(n,t,v)      __declspec(align(n)) t v
#endif

DECLARE_ALIGNED(16,float,ones[4]) = { 1.0f, 1.0f, 1.0f, 1.0f };
DECLARE_ALIGNED(16,float,fours[4]) = { 4.0f, 4.0f, 4.0f, 4.0f };
DECLARE_ALIGNED(16,unsigned,allbits[4]) = {0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};

int readtime()
{
    return SDL_GetTicks();
}

void CoreLoop(double xcur, double ycur, double xstep, unsigned char **p)
{
    DECLARE_ALIGNED(16,float,re[4]);
    DECLARE_ALIGNED(16,float,im[4]);
    DECLARE_ALIGNED(16,unsigned,k1[4]);

#ifdef C_INNER
    DECLARE_ALIGNED(16,float,rez[4]);
    DECLARE_ALIGNED(16,float,imz[4]);
    float t1, t2, o1, o2;
    int k;
#else
    DECLARE_ALIGNED(16,float,outputs[4]);
#endif

    re[0] = (float) xcur;
    re[1] = (float) (xcur + xstep);
    re[2] = (float) (xcur + 2*xstep);
    re[3] = (float) (xcur + 3*xstep);

    im[0] = im[1] = im[2] = im[3] = (float) ycur;

#ifdef C_INNER
    rez[0] = 0.0f;
    rez[1] = 0.0f;
    rez[2] = 0.0f;
    rez[3] = 0.0f;
    imz[0] = 0.0f;
    imz[1] = 0.0f;
    imz[2] = 0.0f;
    imz[3] = 0.0f;

    k = k1[0] = k1[1] = k1[2] = k1[3] = 0;
    while (k < ITERA) {
	if (!k1[0]) {
	    o1 = rez[0] * rez[0];
	    o2 = imz[0] * imz[0];
	    t2 = 2 * rez[0] * imz[0];
	    t1 = o1 - o2;
	    rez[0] = t1 + re[0];
	    imz[0] = t2 + im[0];
	    if (o1 + o2 > 4)
		k1[0] = k;
	}

	if (!k1[1]) {
	    o1 = rez[1] * rez[1];
	    o2 = imz[1] * imz[1];
	    t2 = 2 * rez[1] * imz[1];
	    t1 = o1 - o2;
	    rez[1] = t1 + re[1];
	    imz[1] = t2 + im[1];
	    if (o1 + o2 > 4)
		k1[1] = k;
	}
	
	if (!k1[2]) {
	    o1 = rez[2] * rez[2];
	    o2 = imz[2] * imz[2];
	    t2 = 2 * rez[2] * imz[2];
	    t1 = o1 - o2;
	    rez[2] = t1 + re[2];
	    imz[2] = t2 + im[2];		    
	    if (o1 + o2 > 4)
		k1[2] = k;
	}
	
	if (!k1[3]) {
	    o1 = rez[3] * rez[3];
	    o2 = imz[3] * imz[3];
	    t2 = 2 * rez[3] * imz[3];
	    t1 = o1 - o2;
	    rez[3] = t1 + re[3];
	    imz[3] = t2 + im[3];
	    if (o1 + o2 > 4)
		k1[3] = k;
	}

	if (k1[0]*k1[1]*k1[2]*k1[3])
	    break;

	k++;
    }
    
#else
    k1[0] = k1[1] = k1[2] = k1[3] = 0;

    asm("mov    $0x0,%%ecx\n\t"
	"movaps %4,%%xmm5\n\t"  // fours
	"movaps %2,%%xmm6\n\t"  // re
	"movaps %3,%%xmm7\n\t"  // im
	"xorps  %%xmm0,%%xmm0\n\t"
	"xorps  %%xmm1,%%xmm1\n\t"
	"xorps  %%xmm3,%%xmm3\n\t"
	"1:\n\t"
	"movaps %%xmm0,%%xmm2\n\t"
	"mulps  %%xmm1,%%xmm2\n\t"
	"mulps  %%xmm0,%%xmm0\n\t"
	"mulps  %%xmm1,%%xmm1\n\t"
	"movaps %%xmm0,%%xmm4\n\t"
	"addps  %%xmm1,%%xmm4\n\t"
	"subps  %%xmm1,%%xmm0\n\t"
	"addps  %%xmm6,%%xmm0\n\t"
	"movaps %%xmm2,%%xmm1\n\t"
	"addps  %%xmm1,%%xmm1\n\t"
	"addps  %%xmm7,%%xmm1\n\t"
	"cmpltps %%xmm5,%%xmm4\n\t"
	"movaps %%xmm4,%%xmm2\n\t" //
	"movmskps %%xmm4,%%eax\n\t"
	"andps  %5,%%xmm4\n\t"  // ones
	"addps  %%xmm4,%%xmm3\n\t"
	"or     %%eax,%%eax\n\t"
	"je     2f\n\t"
	"inc    %%ecx\n\t"
	"cmp    $0x77,%%ecx\n\t"
	"jne    1b\n\t"
	"movaps %%xmm2,%%xmm4\n\t"  //
	"xorps  %6,%%xmm4\n\t" // allbits //
	"andps  %%xmm4,%%xmm3\n\t" //
	"2:\n\t"
	"movaps %%xmm3,%0\n\t"
	:"=m"(outputs[0]),"=m"(outputs[2])
	:"m"(re[0]),"m"(im[0]),"m"(fours[0]),"m"(ones[0]),"m"(allbits[0])
	:"%eax","%ecx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","memory");

    k1[0] = (int)(outputs[0]);
    k1[1] = (int)(outputs[1]);
    k1[2] = (int)(outputs[2]);
    k1[3] = (int)(outputs[3]);

#endif

    *(*p)++ = k1[0];
    *(*p)++ = k1[1];
    *(*p)++ = k1[2];
    *(*p)++ = k1[3];
}

void preMandel(double xld, double yld, double xru, double yru)
{
    unsigned char *p = (unsigned char *) previewBuffer;
    double xstep, ystep, xcur, ycur;
    int i, j;

    xstep = (xru - xld)/(MAXX/4);
    ystep = (yru - yld)/(MAXY/4);

#if defined(USE_OPENMP)
    #pragma omp parallel for schedule(dynamic,4) private(p,xcur,ycur,i,j)
#endif
    for (i = 0; i < MAXY/4; i++) {

	xcur = xld;
	ycur = yru - i*ystep;
	p = &previewBuffer[i*MAXX/4];

        for (j = 0; j < MAXX/4; j += 4) {
	    CoreLoop(xcur, ycur, xstep, &p);
	    xcur += 4*xstep;
        }
    }
    for (i = 0; i < MAXY/4; i++) {
	Uint32 offset = i*(MAXX>>2);
	Uint32 firstLeft = 1;
        for (j = 0; j < MAXX/4; j++) {
	    if (previewBuffer[offset+j] == ITERA-1) {
		if (firstLeft) {
		    firstLeft = 0;
		    previewBuffer[offset+j] = 0;
		}
	    } else
		firstLeft = 1;
	}
    }
    for (i = 0; i < MAXY/4; i++) {
	Uint32 offset = i*(MAXX>>2);
	Uint32 firstRight = 1;
        for (j = (MAXX/4) - 1; j>0; j--) {
	    if (previewBuffer[offset+j] == ITERA-1) {
		if (firstRight) {
		    firstRight = 0;
		    previewBuffer[offset+j] = 0;
		}
	    } else
		firstRight = 1;
	}
    }
}

void mandel(double xld, double yld, double xru, double yru)
{
    int i, j;
    double xstep, ystep, xcur, ycur;
    unsigned char *p = (unsigned char *) buffer;

    xstep = (xru - xld)/MAXX;
    ystep = (yru - yld)/MAXY;

#if defined(USE_OPENMP)
#pragma omp parallel for schedule(dynamic,4) private(p,xcur,ycur,i,j)
#endif
    for (i = 0; i < MAXY; i++) {

	Uint32 offset = (i>>2)*(MAXX >> 2);
	xcur = xld;
	ycur = yru - i*ystep;
	p = &buffer[i*MAXX];

        for (j = 0; j < MAXX; j += 4, offset++) {

	    if (ITERA-1 == previewBuffer[offset]) {
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;
		*p++ = 0;

		xcur += 4*xstep;
		continue;
	    }

	    CoreLoop(xcur, ycur, xstep, &p);
	    xcur += 4*xstep;
        }
    }
    showpage();
}

#ifdef _WIN32

#define CHECK(x) {							    \
    unsigned of = (unsigned) &x[0];					    \
    char soThatGccDoesntOptimizeAway[32];				    \
    sprintf(soThatGccDoesntOptimizeAway, "%08x", of);			    \
    if (soThatGccDoesntOptimizeAway[7] != '0') {			    \
	MessageBox(0,							    \
	    "Your compiler is buggy... "				    \
	    "it didn't align the SSE variables...\n"			    \
	    "The application would crash. Aborting.",			    \
	    "Fatal Error", MB_OK | MB_ICONERROR);			    \
	exit(1);							    \
    }									    \
}

#else

#define CHECK(x) {							    \
    unsigned of = (unsigned) &x[0];					    \
    char soThatGccDoesntOptimizeAway[32];				    \
    sprintf(soThatGccDoesntOptimizeAway, "%08x", of);			    \
    if (soThatGccDoesntOptimizeAway[7] != '0') {			    \
	fprintf(stderr,							    \
	    "Your compiler is buggy... "				    \
	    "it didn't align the SSE variables...\n"			    \
	    "The application would crash. Aborting.\n");		    \
	fflush(stderr);							    \
	exit(1);							    \
    }									    \
}

#endif

int main(int argc, char *argv[])
{
    double r = 5.0;
    int i, st, en;
    DECLARE_ALIGNED(16,float,testAlignment[4]);

    CHECK(testAlignment)
    CHECK(ones)
    CHECK(fours)
    CHECK(allbits)

    switch (argc) {
    case 3:
        MAXX = atoi(argv[1]);
        MAXY = atoi(argv[2]);
	MAXX = 16*(MAXX/16);
	MAXY = 16*(MAXY/16);
        break;
    default:
        MAXX = 320;
        MAXY = 240;
        break;
    }

    printf("\nMandelbrot Zoomer by Thanassis (an experiment in SSE).\n\n");
    printf("Stats:\n\t");
#ifdef C_INNER
    printf("(Pipelined floating point calculation)\n\t");
#else
    printf("(SSE calculation)\n\t");
#endif

    previewBuffer = (Uint8 *) malloc(MAXX*MAXY/16);
    init256();

    st = readtime();
    for (i = 0; i < 900; i++, r /= 1.015) {
        preMandel(
		CENTERX - (MAXX*r/MAXY),
		CENTERY - r, 
		CENTERX + (MAXX*r/MAXY), 
		CENTERY + r);
        mandel(
		CENTERX - (MAXX*r/MAXY),
		CENTERY - r, 
		CENTERX + (MAXX*r/MAXY), 
		CENTERY + r);

        if (kbhit())
            break;
    }
    en = readtime();

    close256();

    printf("frames/sec:%5.2f\n\t", ((float) i) / ((en - st) / 1000.0f));
    printf("Zoom factor: %f\n", 1.6 / r);

    return 0;
}

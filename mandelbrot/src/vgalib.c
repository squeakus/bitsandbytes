#include <SDL.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vgalib.h"

unsigned offst[1024];
extern int MAXX, MAXY;
SDL_Surface *surface;
Uint8 *buffer;

void panic(char *fmt, ...)
{
    va_list arg;

    va_start(arg, fmt);
    vfprintf(stderr, fmt, arg);
    va_end(arg);
    exit(0);
}

void init256()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        panic("Couldn't initialize SDL: %d\n", SDL_GetError());
    atexit(SDL_Quit);

    surface = SDL_SetVideoMode(MAXX,
                               MAXY, 8, SDL_HWSURFACE | SDL_HWPALETTE);
    if (!surface)
        panic("Couldn't set video mode: %d", SDL_GetError());

    if (SDL_MUSTLOCK(surface)) {
        if (SDL_LockSurface(surface) < 0)
            panic("Couldn't lock surface: %d", SDL_GetError());
    }
    buffer = (Uint8*)surface->pixels;
    {
        int i;
        for (i = 0; i < MAXY; i++)
            offst[i] = i * MAXX;
    }

    if (1) {
        SDL_Color palette[256];
        int i;
	int ofs=0;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 16*(16-abs(i-16));
            palette[i+ofs].g = 0;
            palette[i+ofs].b = 16*abs(i-16);
        }
	ofs= 16;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 0;
            palette[i+ofs].g = 16*(16-abs(i-16));
            palette[i+ofs].b = 0;
        }
	ofs= 32;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 0;
            palette[i+ofs].g = 0;
            palette[i+ofs].b = 16*(16-abs(i-16));
        }
	ofs= 48;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 16*(16-abs(i-16));
            palette[i+ofs].g = 16*(16-abs(i-16));
            palette[i+ofs].b = 0;
        }
	ofs= 64;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 0;
            palette[i+ofs].g = 16*(16-abs(i-16));
            palette[i+ofs].b = 16*(16-abs(i-16));
        }
	ofs= 80;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 16*(16-abs(i-16));
            palette[i+ofs].g = 0;
            palette[i+ofs].b = 16*(16-abs(i-16));
        }
	ofs= 96;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 16*(16-abs(i-16));
            palette[i+ofs].g = 16*(16-abs(i-16));
            palette[i+ofs].b = 16*(16-abs(i-16));
        }
	ofs= 112;
        for (i = 0; i < 16; i++) {
            palette[i+ofs].r = 16*(8-abs(i-8));
            palette[i+ofs].g = 16*(8-abs(i-8));
            palette[i+ofs].b = 16*(8-abs(i-8));
        }
        SDL_SetColors(surface, palette, 0, 256);
    }
}

void close256()
{
}


void qplot(int r, int k, int l)
{
    *(buffer + offst[k] + r) = l;
}

int kbhit()
{
    SDL_Event event;

    if (SDL_PollEvent(&event)) {
        if (event.type == SDL_KEYDOWN)
            return 1;
    }
    return 0;
}


void showpage(void)
{
    SDL_UpdateRect(surface, 0, 0, MAXX, MAXY);
}

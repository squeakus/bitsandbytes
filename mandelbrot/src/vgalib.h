#ifndef __VGALIB_H__
#define __VGALIB_H__

/* Ah, the old days... 
 * this is the "API" I had when I coded in ASM... 
 * vgalib.c does the same using SDL :-) */

void init256();
void close256();
void qplot(int r,int k,int l);
void pal(void);
void showpage(void);
int kbhit(void);

#endif

/* medit.c      mesh visualization tool
 *
 * Written by Pascal Frey, LJLL
 * Copyright (c) Inria, 1999-2007. All rights reserved. */

#include "medit.h"
#ifdef ppc
#include <unistd.h>
#endif
#include "GL/osmesa.h"
#include "GL/glu.h"

/* global variables (see extern.h) */
void *buffer;
GLboolean hasStereo = 1;
int refitem = 1;
int refmat=1;
Canvas    cv;
ubyte     ddebug,animate,saveimg,imgtype,infogl,fullscreen;
ubyte     quiet,option,morphing,stereoMode;
int       menu,amenu,fmenu,femenu,vmenu,mmenu,smenu;
int       clmenu,cmenu,vwmenu,txmenu,trmenu;
int       animdep,animfin;
int WIDTH = 600;
int HEIGHT = 600;
int bits = 8;
GLenum type = GL_UNSIGNED_BYTE;

static void excfun(int sigid) {
  fprintf(stdout,"\n Unexpected error:");  fflush(stdout);
  switch(sigid) {
    case SIGABRT:
      fprintf(stdout,"  Abnormal stop\n");  break;
    case SIGFPE:
      fprintf(stdout,"  Floating-point exception\n"); break;
    case SIGILL:
      fprintf(stdout,"  Illegal instruction\n"); break;
    case SIGSEGV:
      fprintf(stdout,"  Segmentation fault\n");  break;
    case SIGTERM:
    case SIGINT:
      fprintf(stdout,"  Program killed\n");  break;
  }
  exit(1);
}

static void endcod() {
  fprintf(stdout," Thank you for using Medit.\n");
}


static void grInfo(void) {
  GLboolean  b;
  GLint      i,win;
 
  fprintf(stdout,"Graphic info:\n");
  fprintf(stdout," GL Vendor:\t%s\n",glGetString(GL_VENDOR));
  fprintf(stdout," GL Version:\t%s\n",glGetString(GL_VERSION));
  fprintf(stdout," GL Renderer:\t%s\n\n",glGetString(GL_RENDERER));
  glGetBooleanv(GL_RGBA_MODE,&b);
  if ( b )  fprintf(stdout,"  RGBA Mode\n");
  glGetBooleanv(GL_DOUBLEBUFFER,&b);
  if ( b )  fprintf(stdout,"  Double Buffer\n");
  glGetBooleanv(GL_STEREO,&b);
  if ( b )  fprintf(stdout,"  Stereo\n");
  glGetIntegerv(GL_AUX_BUFFERS,&i);
  if ( i )  fprintf(stdout,"  Auxilary Buffers\t%2d\n",(int)i);
  glGetIntegerv(GL_INDEX_BITS,&i);
  if ( i )  fprintf(stdout,"  Index Bits\t\t%2d\n",(int)i);
  glGetIntegerv(GL_RED_BITS,&i);
  fprintf(stdout,"  RGBA Bits\t\t%2d",(int)i);
  glGetIntegerv(GL_GREEN_BITS,&i);
  fprintf(stdout,"\t%2d",(int)i);
  glGetIntegerv(GL_BLUE_BITS,&i);
  fprintf(stdout,"\t%2d",(int)i);
  glGetIntegerv(GL_ALPHA_BITS,&i);
  fprintf(stdout,"\t%2d\n",(int)i);
  glGetIntegerv(GL_ACCUM_RED_BITS,&i);
  fprintf(stdout,"  Accum RGBA Bits\t%2d",(int)i);
  glGetIntegerv(GL_ACCUM_GREEN_BITS,&i);
  fprintf(stdout,"\t%2d",(int)i);
  glGetIntegerv(GL_ACCUM_BLUE_BITS,&i);
  fprintf(stdout,"\t%2d",(int)i);
  glGetIntegerv(GL_ACCUM_ALPHA_BITS,&i);
  fprintf(stdout,"\t%2d\n",(int)i);
  glGetIntegerv(GL_DEPTH_BITS,&i);
  fprintf(stdout,"  Depth Bits\t\t%2d\n",(int)i);
  glGetIntegerv(GL_STENCIL_BITS,&i);
  fprintf(stdout,"  Stencil Bits\t\t%2d\n",(int)i);
  
  exit(1);
}

int medit0() {
  pMesh    mesh;
  char     data[128],*name;
  int      k,l,ret;
  clock_t  ct;

  /* default */
  fprintf(stdout," loading data file(s)\n");
  ct = clock();

  /* enter name */
  if ( !cv.nbm ) {
    fprintf(stdout,"  File name(s) missing. Please enter : ");
    fflush(stdout); fflush(stdin);
    fgets(data,120,stdin);
    if ( !strlen(data) ) {
      fprintf(stdout,"  ## No data\n");
      return(0);
    }

    /* parse file name(s) */
    fprintf(stdout,"data: %s",data);
    name = strtok(data," \n");
    while( name ) {
      if ( !cv.mesh[cv.nbm] ) {
        cv.mesh[cv.nbm] = (pMesh)M_calloc(1,sizeof(Mesh),"medit0.mesh");
        if ( !cv.mesh[cv.nbm] )  return(0);
      }
      strcpy(cv.mesh[cv.nbm]->name,name);
      name = strtok(NULL," \n\0");
      if ( ++cv.nbm == MAX_MESH )  break;
    }
    if ( !cv.nbm ) return(0);
  }

  /* read mesh(es)*/
  k = 0;
  do {
    if ( !cv.mesh[k] ) {
      cv.mesh[k] = M_calloc(1,sizeof(Mesh),"medit0.mesh");
      if ( !cv.mesh[k] )  return(0);
    }
    mesh = cv.mesh[k];
    mesh->typ = 0;
    ret  = loadMesh(mesh);
  
    if ( ret <= 0 ) {
      for (l=k+1; l<cv.nbm; l++)
	    cv.mesh[l-1] = cv.mesh[l];
      cv.nbm--;
      k--;
      continue;
    }
    /* compute mesh box */
    if ( (mesh->ntet && !mesh->nt) || (mesh->nhex && !mesh->nq) )  
      meshSurf(mesh);
    meshBox(mesh,1);
    if ( !quiet )  meshInfo(mesh);

    /* read metric */
    /*if ( !loadSol(mesh,mesh->name,1) )
      bbfile(mesh);*/
    if ( !quiet && mesh->nbb )
      fprintf(stdout,"    Solutions  %8d\n",mesh->nbb);
  }
  while ( ++k < cv.nbm );
  cv.nbs = cv.nbm;

  ct = difftime(clock(),ct);
  fprintf(stdout,"  Input seconds:     %.2f\n",
          (double)ct/(double)CLOCKS_PER_SEC);

  return(cv.nbm);
}


int medit1() {
  pScene   scene;
  pMesh    mesh;
  int      k;
  clock_t  ct;

  /* create grafix */
  fprintf(stdout,"\n Building scene(s)\n");
  ct = clock();
  for (k=0; k<cv.nbs; k++) {
    if ( !cv.scene[k] ) {
      cv.scene[k] = (pScene)M_calloc(1,sizeof(Scene),"medit1.scene");
      if ( !cv.scene[k] )  return(0);
    }
    scene = cv.scene[k];
    if ( !cv.mesh[k] ) {
      cv.mesh[k] = (pMesh)M_calloc(1,sizeof(Mesh),"medit1.mesh");
      if ( !cv.mesh[k] )  return(0);
    }
    mesh  = cv.mesh[k];

    fprintf(stdout,"  Creating scene %d\n",k+1);
    parsop(scene,mesh);
    meshRef(scene,mesh);
    matSort(scene);
	
   if ( !createScene(scene,k) ) {
      fprintf(stderr,"  ## Unable to create scene\n");
      return(0);
    }
  }
  ct = difftime(clock(),ct);
  fprintf(stdout,"  Scene seconds:     %.2f\n",(double)ct/(double)CLOCKS_PER_SEC);
  return(1);
}


int main(int argc,char *argv[]) {

  char   pwd[1024];


#ifdef ppc
  if ( !getwd(pwd) )  exit(2);
#endif

  fprintf(stdout,"  -- Medit,  Release %s (%s)\n",ME_VER,ME_REL);
  fprintf(stdout,"     %s.\n",ME_CPY);

  /* trap exceptions */
  signal(SIGABRT,excfun);
  signal(SIGFPE,excfun);
  signal(SIGILL,excfun);
  signal(SIGSEGV,excfun);
  signal(SIGTERM,excfun);
  signal(SIGINT,excfun);
  atexit(endcod);

  /* default values */
  option     = SCHNAUZER;
  saveimg    = GL_TRUE;
  imgtype    = P6;
  animate    = FALSE;
  morphing   = FALSE;
  fullscreen = FALSE;
  animdep    = 0;
  animfin    = 0;
  ddebug     = FALSE;
  quiet      = FALSE;
  stereoMode = 0;
  cv.nbm = cv.nbs = 0;
  /* init grafix */
  parsar(argc,argv);

#ifdef ppc
  chdir(pwd);
#endif

   /*glutInit(&argc,argv);*/
   const GLint z = 16, stencil = 0, accum = 0;
   OSMesaContext ctx;
   GLint cBits;

   assert(bits == 8);
   assert(type == GL_UNSIGNED_BYTE);

   ctx = OSMesaCreateContextExt(OSMESA_RGBA, z, stencil, accum, NULL );
   if (!ctx) {
      printf("OSMesaCreateContextExt() failed!\n");
      return 0;
   }

   /* Allocate the image buffer */
   buffer = malloc(WIDTH * HEIGHT * 4 * bits / 8);
   if (!buffer) {
      printf("Alloc image buffer failed!\n");
      return 0;
   }

   /* Bind the buffer to the context and make it current */
   if (!OSMesaMakeCurrent( ctx, buffer, type, WIDTH, HEIGHT )) {
      printf("OSMesaMakeCurrent (%d bits/channel) failed!\n", bits);
      free(buffer);
      OSMesaDestroyContext(ctx);
      return 0;
   }

   /* sanity checks */
   glGetIntegerv(GL_RED_BITS, &cBits);
   printf("%d\n", cBits);
   assert(cBits == bits);
   glGetIntegerv(GL_GREEN_BITS, &cBits);
   assert(cBits == bits);
   glGetIntegerv(GL_BLUE_BITS, &cBits);
   assert(cBits == bits);
   glGetIntegerv(GL_ALPHA_BITS, &cBits);
   assert(cBits == bits);

  if ( infogl )  grInfo();
  
  /* call animate or normal mode */
  switch (option) {
  case SCHNAUZER:
    if ( ddebug )  printf("Using Schnauzer\n");
    if ( !medit0() )  exit(1);
    if ( !medit1() )  exit(1);
    reshapeScene(WIDTH, HEIGHT);
    redrawSchnauzer();
    break;
  case SEQUENCE:
    if ( ddebug )  printf("Using Animation\n");
    if ( !animat() )  exit(1);
    redrawSequence();
    break;

  default:
    fprintf(stderr,"  ## Unrecognized option %d\n",option);
    exit(1);
    break;
  }

  /* main grafix loop */
  fprintf(stdout,"\n *************************************\n");
  glGetBooleanv(GL_STEREO,&hasStereo);
  return(0);
}

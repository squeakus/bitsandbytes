#include "medit.h"
#include "extern.h"
#include "sproto.h"

extern GLboolean  hasStereo;
extern int       *pilmat,ipilmat,refmat,reftype,refitem;
extern short      schw,schh;
extern ubyte      quiet,fullscreen,tiling,stereoMode;

/* return current active scene */
int currentScene() {
  int  k,idw;

  for (k=0; k<MAX_SCENE; k++) {
    if ( cv.scene[k]) /* && idw == cv.scene[k]->idwin)*/
      return(k);
  }
  return(0);
}

/* check for OpenGL error */
void checkErrors(void) {
  GLenum error;

  while ( (error = glGetError()) != GL_NO_ERROR ) {
    fprintf(stderr,"  ## ERROR: %d: %s\n",
            (int)error,(char*)gluErrorString(error));
    exit(1);
  }
}

void farclip(GLboolean b) {
  pScene  sc;
  pPersp  p;
  pCamera c;
  float   look[3],ratio,units;
  int     idw = currentScene();
  static  GLfloat up[3] = { 0.0, 1.0, 0.0};

  /* default */
  sc    = cv.scene[idw];
  p     = sc->persp;
  c     = sc->camera;
  ratio = (GLfloat)sc->par.xs / sc->par.ys;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  switch (p->pmode) {
  case ORTHO:
    glOrtho(-1.,1.,-1.,0.1,0.01,0.01);
    break;

  case PERSPECTIVE:
    if ( b )
      gluPerspective(p->fovy,ratio,sc->dmax,4.0*sc->dmax);
    else
      gluPerspective(p->fovy,ratio,0.01,10000.0*sc->dmax);
    units = 1.e-02;
    glPolygonOffset(1.0, units);
    break;

  case CAMERA:
    gluPerspective(p->fovy,ratio,0.001*sc->dmax,4.0*sc->dmax);
    look[0] = c->eye[0] + 0.001*sc->dmax*c->speed[0];
    look[1] = c->eye[1] + 0.001*sc->dmax*c->speed[1];
    look[2] = c->eye[2] + 0.001*sc->dmax*c->speed[2];
    gluLookAt(c->eye[0],c->eye[1],c->eye[2], 
              look[0],look[1],look[2],
              up[0],up[1],up[2]);
    break;
  }

  /* zoom transformation */
  if ( p->rubber == 2 ) {
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-p->gamma,1.,0.,0.);
    glRotatef(p->alpha,0.,1.,0.);
    glMultMatrixf(p->matrix);
    glGetFloatv(GL_PROJECTION_MATRIX,p->matrix);
    glPopMatrix();
    p->rubber = 0;
  }

  /* apply transformation */
  glMultMatrixf(p->matrix);
  glMatrixMode(GL_MODELVIEW);
}

void reshapeScene(int width,int height) {
  pScene   sc;

  if ( ddebug ) fprintf(stdout,"reshape scene %d\n",currentScene());
  sc = cv.scene[currentScene()];
  sc->par.xs = width;
  sc->par.ys = height;

  glViewport(0,0,width,height);
  farclip(GL_TRUE);
  }

static void drawList(pScene sc,int clip,int map) {
  pMesh   mesh = cv.mesh[sc->idmesh];
  ubyte   elev = sc->mode & S_ALTITUDE;

  if ( ddebug ) printf("drawList %p %d %d\n",sc,clip,map);
  if ( mesh->dim == 2 && !elev ) glDisable(GL_DEPTH_TEST);

  glLineWidth(1.0);
  if ( clip ) {
    if ( map ) {
      if ( sc->cmlist[LTets] ) glCallList(sc->cmlist[LTets]);
      if ( sc->cmlist[LHexa] ) glCallList(sc->cmlist[LHexa]);
    }
    else {
      if ( sc->clist[LTets] ) glCallList(sc->clist[LTets]);
      if ( sc->clist[LHexa] ) glCallList(sc->clist[LHexa]);
    }
  }
  else if ( map ) {
    if ( mesh->nt+mesh->nq ) {
      if ( sc->mlist[LTria] ) glCallList(sc->mlist[LTria]);
      if ( sc->mlist[LQuad] ) glCallList(sc->mlist[LQuad]);
    }
    else {
      if ( sc->mlist[LTets] ) glCallList(sc->mlist[LTets]);
      if ( sc->mlist[LHexa] ) glCallList(sc->mlist[LHexa]);
    }
  }
  else {
    if ( mesh->nt+mesh->nq ) {
      if ( sc->dlist[LTria] ) glCallList(sc->dlist[LTria]);
      if ( sc->dlist[LQuad] ) glCallList(sc->dlist[LQuad]);
    }
    else {
      if ( sc->dlist[LTets] ) glCallList(sc->dlist[LTets]);
      if ( sc->dlist[LHexa] ) glCallList(sc->dlist[LHexa]);
    }
  }
  if ( mesh->dim == 2 && !elev ) glEnable(GL_DEPTH_TEST);
}

#ifdef ppc
void bogusQuad(pScene sc) {
  /* bogus polygon (nvidia) */
  glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glLineWidth(1.0);
    glColor3fv(sc->par.back);
    glBegin(GL_QUADS);
      glVertex3f(0., 0.,-sc->persp->depth);
      glVertex3f(0., 0.,-sc->persp->depth);
      glVertex3f(0., 0.,-sc->persp->depth);
      glVertex3f(0., 0.,-sc->persp->depth);
    glEnd();
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
}
#endif

static void displayScene(pScene sc,int mode,int clip) {
  int     map;
  
  map = mode & S_MAP;

  switch(mode) {
  case FILL:  /* solid fill */
    if ( ddebug ) printf("solid fill\n");
    glEnable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glDisable(GL_POLYGON_OFFSET_FILL);
      drawList(sc,clip,0);
    glDisable(GL_LIGHTING);
    break;

  case WIRE:  /* basic wireframe */
  case WIRE+S_MATERIAL:
    if ( ddebug ) printf("wireframe\n");
#ifdef ppc
    bogusQuad(sc);
#endif
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor4fv(sc->par.line);
    glDisable(GL_POLYGON_OFFSET_FILL);
      drawList(sc,clip,0);
    break;

  case DEPTH:  /* depth wireframe */
  case DEPTH + S_MATERIAL:
    if ( ddebug ) printf("depth wireframe\n");
    glEnable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
      drawList(sc,clip,0);
    glDisable(GL_LIGHTING);
    break;
  case HIDDEN: /* hidden lines removal */
  case HIDDEN + S_MATERIAL:
    if ( ddebug ) printf("hidden lines\n");
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glColor3fv(sc->par.back);
      drawList(sc,clip,0);
    glDisable(GL_POLYGON_OFFSET_FILL);
#ifdef ppc
    bogusQuad(sc);
#endif
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor4fv(sc->par.line);
      drawList(sc,clip,0);
    break;



  default: /* other modes */
    if ( ddebug ) printf("rendering mode %d\n",sc->mode);
    /* interior */
    if ( sc->mode & S_FILL ) {
      if ( sc->mode & S_COLOR )  glEnable(GL_LIGHTING);
      glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
      glEnable(GL_POLYGON_OFFSET_FILL);
      if ( sc->mode & S_MAP ) {
        glEnable(GL_COLOR_MATERIAL);
      drawList(sc,clip,map);
        glDisable(GL_COLOR_MATERIAL);
      }
      else {
        glColor4fv(sc->par.back);
        drawList(sc,clip,0);
      }
    }

    /* boundary */
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_POLYGON_OFFSET_FILL);
#ifdef ppc
    bogusQuad(sc);
#endif
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    if ( !(sc->mode & S_BDRY) )  break;
    if ( sc->mode & S_COLOR && !(sc->mode & S_FILL) ) {
      glEnable(GL_COLOR_MATERIAL);
      glEnable(GL_LIGHTING);
    }
    if ( sc->mode & S_MAP) {
      if ( sc->mode & S_FILL ) {
        glColor4fv(sc->par.line);
        drawList(sc,clip,0);
      }
      else
        drawList(sc,clip,map);
    }
    else if ( sc->mode & S_ALTITUDE ) {
      glColor4fv(sc->par.line);
      drawList(sc,clip,map);
      if ( sc->mlist[LTets] ) glCallList(sc->mlist[LTets]);
      if ( sc->mlist[LHexa] ) glCallList(sc->mlist[LHexa]);
    }
    else {
      glColor4fv(sc->par.line);
      drawList(sc,clip,0);
    }
  }
}


void setupView(pScene sc) {
  pScene       slave;
  pMesh        mesh;
  pTransform   view;
  pPersp       p;
  pCamera      c;
  int          clvol;

  /* default */
  if ( ddebug )  fprintf(stdout,"setupView\n");
  mesh = cv.mesh[sc->idmesh];
  view = sc->view;
  p    = sc->persp;
  c    = sc->camera;

  /* init transformation matrix */
  if ( sc->type & S_RESET ) {
    glPushMatrix();
    glLoadIdentity();
    if ( p->pmode != CAMERA ) {
      if ( mesh->dim == 3 || sc->mode & S_ALTITUDE ) {
        glRotatef(-60.,1.,0.,0.);
        glRotatef(-120.,0.,0.,1.);
      }
    }
    else {
      if ( c->vecup == X_AXIS )
        glRotatef(90.0,0.0,0.0,1.0);
      else if ( c->vecup == Z_AXIS )
        glRotatef(-90.0,1.0,0.0,0.0);
    }
    glGetFloatv(GL_MODELVIEW_MATRIX,view->matrix);
    glPopMatrix();
    sc->type ^= S_RESET;
  }

  /* keep old transformation */
  memcpy(view->oldmat,view->matrix,16*sizeof(float));

  /* compute new transformation */
  glPushMatrix();
  glLoadIdentity();
  if ( p->pmode != CAMERA ) {
    glTranslatef(view->panx,view->pany,0.0);
    if ( mesh->dim == 3 || sc->mode & S_ALTITUDE )
      glRotatef(view->angle,view->axis[0],view->axis[1],view->axis[2]);
    glTranslatef(-view->opanx,-view->opany,0.);
    glMultMatrixf(view->matrix);
    glGetFloatv(GL_MODELVIEW_MATRIX,view->matrix);
  }
  glPopMatrix();

  /* keep old translation */
  view->opanx = view->panx;
  view->opany = view->pany;

  /* copy views */
  if ( !animate && sc->slave > -1 ) {
    slave = cv.scene[sc->slave];
    memcpy(slave->view,sc->view,sizeof(struct transform));
    memcpy(slave->camera,sc->camera,sizeof(struct camera));
    slave->view->angle = 0.0f;
    clvol = slave->clip->active & C_VOL;
    memcpy(slave->clip,sc->clip,sizeof(struct clip));
    if ( clvol )  slave->clip->active |= C_VOL;
  }
}

void drawModel(pScene sc) {
  pMesh        mesh;
  pTransform   view;
  pClip        clip;
  ubyte        sstatic;

  /* default */
  mesh = cv.mesh[sc->idmesh];
  view = sc->view;
  clip = sc->clip;
  if ( ddebug ) printf("\n-- redraw scene %d, mesh %d\n",sc->idwin,sc->idmesh);
  glDisable(GL_LIGHTING);

  /* draw clipping plane */
  if ( clip->active & C_ON ) {
    drawClip(sc,clip,mesh,0);
    glClipPlane(GL_CLIP_PLANE0,clip->eqn);
    glEnable(GL_CLIP_PLANE0);
  }
  else
    glDisable(GL_CLIP_PLANE0);

  /* draw object if static scene */
  sstatic = view->mstate > 0 && clip->cliptr->mstate > 0;
  if ( sstatic || sc->type & S_FOLLOW ) {
    displayScene(sc,sc->mode,0);
    if ( sc->item & S_NUMP || sc->item & S_NUMF )  listNum(sc,mesh);
    /* draw normals */
    if ( sc->type & S_NORMAL ) {
      if ( !sc->nlist ) sc->nlist = drawNormals(mesh,sc);
      glCallList(sc->nlist);
    }
  }

  /* draw ridges, corners, etc. */
    if ( (sc->item & S_GEOM) && sc->glist ) {
    glDisable(GL_LIGHTING);
    if ( !mesh->ne )
      glPointSize(1);
    else
      glPointSize(5);
    glDisable(GL_COLOR_MATERIAL);
    glCallList(sc->glist);
    }

  drawBase(sc,mesh);
  drawBox(sc,mesh,0);
  glDisable(GL_CLIP_PLANE0);
  /*if ( (mesh->dim == 3 || sc->mode & S_ALTITUDE) && sc->item & S_GRID )*/
  if ( sstatic && clip->active & C_ON && clip->active & C_VOL )
    displayScene(sc,sc->mode,1);

  /* show path, if any */
  if ( sc->type & S_PATH && sc->path.tlist )
    glCallList(sc->path.tlist);
}

void drawBox(pScene sc,pMesh mesh,int mode) {
  pMaterial  pm;
  float      cx,cy,cz;
  int        i,k,m;

  glDisable(GL_LIGHTING);
  glPushMatrix();
  glScalef(1.01 * fabs(mesh->xmax-mesh->xmin),
           1.01 * fabs(mesh->ymax-mesh->ymin),
           1.01 * fabs(mesh->zmax-mesh->zmin));
  glColor3f(1.0,0.0,0.5);
  /*glutWireCube(1.0);*/
  glPopMatrix();

  /* one box per sub-domain */
  if ( mode ) {
    for (m=0; m<sc->par.nbmat; m++) {
      pm = &sc->material[m];
      for (i=0; i<MAX_LIST; i++) {
        k  = pm->depmat[i];
        if ( !k || pm->flag )  continue;
        cx = 0.5 * (pm->ext[3]+pm->ext[0]);
        cy = 0.5 * (pm->ext[4]+pm->ext[1]);
        cz = 0.5 * (pm->ext[5]+pm->ext[2]);
        glPushMatrix();
        glColor3fv(pm->dif);
        glTranslatef(cx,cy,cz);
        glScalef(pm->ext[3]-pm->ext[0],pm->ext[4]-pm->ext[1],
                 pm->ext[5]-pm->ext[2]);
        /*glutWireCube(1.0);*/
        glPopMatrix();
      }
    }
  }
}

void drawBase(pScene sc,pMesh mesh) {
  int  k;

  /* default */
  if ( ddebug ) printf("draw base\n");

  if ( !sc->grid ) {
    sc->grid = glGenLists(1);
    glNewList(sc->grid,GL_COMPILE);
    if ( glGetError() )  return;
    glColor3f(0.5,0.5,0.5);
    glLineWidth(2.0);
    glBegin(GL_LINES);
    for (k=0; k<21; k+=5) {
      glVertex3f(k*0.05,0.,0.);  glVertex3f(k*0.05,1.,0.);
      glVertex3f(0.,k*0.05,0.);  glVertex3f(1.,k*0.05,0.);
    }
    glEnd();
    glColor3f(0.6,0.6,0.6);
    glLineWidth(1.0);
    glBegin(GL_LINES);
    for (k=0; k<21; k++) {
      if ( k%5 == 0 ) continue;
      glVertex3f(k*0.05,0.,0.);  glVertex3f(k*0.05,1.,0.);
      glVertex3f(0.,k*0.05,0.);  glVertex3f(1.,k*0.05,0.);
    }
    glEnd();
    glEndList();
  }

  glPushMatrix();
  glTranslatef(-1.5*sc->dmax,-1.5*sc->dmax,-0.5*(mesh->zmax-mesh->zmin));
  glScalef(3*sc->dmax,3*sc->dmax,3*sc->dmax);
  glDisable(GL_LIGHTING);
    glCallList(sc->grid);
  glPopMatrix();
}


/* OpenGL callbacks */
void redrawScene() {
  pScene       sc,slave;
  pTransform   view;
  pPersp       p;
  pCamera      c;
  double       ndfl,ratio,top,bottom,left,right,nnear,ffar;
  reshapeScene(WIDTH, HEIGHT);
  if (ddebug) fprintf(stdout,"redrawing scene\n");
  sc   = cv.scene[currentScene()];
  view = sc->view;
  p    = sc->persp;
  c    = sc->camera;

  if ( stereoMode == MONO || !hasStereo ) {
    glDrawBuffer(GL_BACK_LEFT);
    glClearColor(sc->par.back[0],sc->par.back[1],
                 sc->par.back[2],sc->par.back[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.,0.,-p->depth, 0.,0.,0., 0.0,1.0,0.0);

    setupView(sc);
    glMultMatrixf(view->matrix);
    glTranslatef(sc->cx,sc->cy,sc->cz);
    drawModel(sc);
  }

  else {
    nnear   = -p->depth - 0.5 * sc->dmax;
    if ( nnear < 0.1 )  nnear = 0.1;
    ffar    = -p->depth + 0.5 * sc->dmax;
    ratio   = sc->par.xs / (double)sc->par.ys;
    top     = nnear * tan(DTOR * 0.5 * p->fovy);
    ndfl    = nnear / p->depth;
    if ( sc->par.eyesep < 0.0 )
      sc->par.eyesep = fabs(p->depth / 20.0);

    /* left view */
    glDrawBuffer(GL_BACK_LEFT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    left  = -ratio * top + 0.5 * sc->par.eyesep * ndfl;
    right =  ratio * top + 0.5 * sc->par.eyesep * ndfl;
    bottom= -top;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left,right,top,bottom,nnear,ffar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(-sc->par.eyesep,0.,-p->depth, 
              sc->par.eyesep/3.0,0.,0., 
              0.0,1.0,0.0);

    setupView(sc);
    glMultMatrixf(view->matrix);
    glTranslatef(sc->cx,sc->cy,sc->cz);
    drawModel(sc);

    /* right view */
    glDrawBuffer(GL_BACK_RIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    left  = -ratio * top - 0.5 * sc->par.eyesep * ndfl;
    right =  ratio * top - 0.5 * sc->par.eyesep * ndfl;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left,right,top,bottom,nnear,ffar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(sc->par.eyesep,0.,-p->depth, 
              sc->par.eyesep/3.0,0.,0., 
              0.0,1.0,0.0);

    setupView(sc);
    glMultMatrixf(view->matrix);
    glTranslatef(sc->cx,sc->cy,sc->cz);
    drawModel(sc);
  }

  /* refresh screen */
  if ( saveimg && animate )
    glFlush();
  else
    glFlush();
  if ( saveimg )  saveHard();
}

void saveHard() {
  pScene       sc;
  pMesh        mesh;
  pTransform   view;
  char        *ptr,data[128];

  /* default */
  if ( ddebug ) fprintf(stdout,"saving scene\n");
  sc   = cv.scene[currentScene()];
  mesh = cv.mesh[sc->idmesh];
  view = sc->view;

  strcpy(data,mesh->name);
  ptr = (char*)strstr(data,".mesh");
  if ( ptr ) *ptr = '\0';
  strcat(data,".ppm");
  imgHard(sc,data,'H');
}

void redrawSequence() {
  fprintf(stdout,"redrawing sequence\n");
  pScene  sc;
  pMesh   mesh;
  pClip   clip;
  static int depart = -1;

  /* default */
  sc   = cv.scene[currentScene()];
  mesh = cv.mesh[sc->idmesh];
  clip = sc->clip;
  /*redrawScene();
    imgHard(sc,"moo.ppm",'H');*/

  if ( depart == -1 )
    depart = animdep;
    if ( ddebug ) fprintf(stdout,"debut sequence %d a %d\n",animdep,animfin);
  if ( option == SEQUENCE )
    if ( ddebug ) fprintf(stdout,"calling playAnim\n");
    playAnim(sc,mesh,animdep,animfin);
  exit(0);
}


/* OpenGL callbacks */
void redrawSchnauzer() {
  fprintf(stdout,"redrawing that schnauzer\n");
  pScene  sc = cv.scene[currentScene()];
  pMesh   mesh;
  char   *ptr,data[256];

  mesh = cv.mesh[sc->idmesh];
  strcpy(data,mesh->name);
  ptr = (char*)strstr(data,".mesh");
  if ( ptr ) *ptr = '\0';
  strcat(data,".ppm");
  redrawScene();
  /*imgHard(sc,data,'H');*/
  exit(0);
}

void initGrafix(pScene sc,pMesh mesh) {
  GLfloat  lightamb[4] = { 0.3, 0.3, 0.3, 1.0 };
  GLfloat  lightdif[4] = { 1.0, 1.0, 1.0, 1.0 };
  GLfloat  lightpos[4] = { 0.0, 0.0, 1.0, 0.0 };

  if ( ddebug )  printf("initGrafix\n");
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glPolygonOffset(1.0, 1.0 / (float)0x10000);
  glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);

  glShadeModel(GL_SMOOTH);
  glDisable(GL_NORMALIZE);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_DITHER);
  glDisable(GL_CULL_FACE);
  if ( mesh->typ == 2 ) {
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);

  /* lighting */
  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_FALSE);
#if ( !defined(GL_VERSION_1_1) )
  glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL,GL_SEPARATE_SPECULAR_COLOR);
#endif

  glLightfv(GL_LIGHT0,GL_DIFFUSE,lightdif);
  glLightfv(GL_LIGHT0,GL_AMBIENT,lightamb);
  glEnable(GL_LIGHTING);
  if ( stereoMode != MONO ) {
    lightpos[2] = -1.0;
    if ( sc->par.sunpos )
      sc->par.sunpos[2] = -fabs(sc->par.sunpos[2]);
  }
  if ( sc->par.sunp )
    glLightfv(GL_LIGHT0,GL_POSITION,sc->par.sunpos);
  else
    glLightfv(GL_LIGHT0,GL_POSITION,lightpos);
  glEnable(GL_LIGHT0);
}


/* new scene */
int createScene(pScene sc,int idmesh) {
  pMesh     mesh;
  char      data[128];

  /* default */
  mesh = cv.mesh[idmesh];
  if ( ddebug ) fprintf(stdout,"   Computing 3D scene\n");
  sc->item ^= S_GRID;

  /* set default mode */
  sc->idmesh = idmesh;
  sc->par.xi = sc->par.yi = 10;
  if ( option == SCHNAUZER ) {
    sc->par.xs = schw;
    sc->par.ys = schh;
  }
  else {
    if ( sc->par.xs == 0 )  sc->par.xs = 600;
    if ( sc->par.ys == 0 )  sc->par.ys = 600;
  }
  if ( !sc->mode )  sc->mode = HIDDEN;

  sc->item   = 0;
  sc->shrink = 1.0;
  sc->slave  = sc->master = -1;
  sc->picked = 0;
  if ( mesh->nvn == 0 )  sc->type ^= S_FLAT;
  if ( mesh->ne == 0 )   sc->item |= S_GEOM;

  /* compute scene depth */
  sc->dmax = sc->dmin= mesh->xmax - mesh->xmin;
  sc->dmax = max(sc->dmax,mesh->ymax - mesh->ymin);
  sc->dmin = min(sc->dmin,mesh->ymax - mesh->ymin);
  if ( mesh->dim == 3 ) {
    sc->dmax = max(sc->dmax,mesh->zmax - mesh->zmin);
    sc->dmin = min(sc->dmin,mesh->zmax - mesh->zmin);
  }
  sc->dmax = fabs(sc->dmax);
  sc->dmin = fabs(sc->dmin);
  if ( !sc->par.sunp ) {
    sc->par.sunpos[0] *= 2.0*sc->dmax;
    sc->par.sunpos[1] *= 2.0*sc->dmax;
    sc->par.sunpos[2] *= 2.0*sc->dmax;
  }
  sc->par.sunpos[3] = 1.0;

 /* required! to change background color */
  glClearColor(sc->par.back[0],sc->par.back[1],
               sc->par.back[2],sc->par.back[3]);

  /* init perspective */
  sc->persp  = initPersp(0,sc->dmax);
  sc->camera = (pCamera)initCamera(sc,Y_AXIS);
  if ( mesh->typ == 2 ) {
    sc->persp->pmode = CAMERA;
    sc->persp->depth *= 0.5;
  }

  /* create default view */
  sc->view = (pTransform)createTransform();
  if ( !sc->view )  return(0);
  sc->type |= S_RESET + S_DECO;
  sc->clip  = (pClip)createClip(sc,mesh);
  if ( !sc->clip )  return(0);
  sc->cube  = (pCube)createCube(sc,mesh);
  if ( !sc->cube )  return(0);

  /* create display lists by geom type */
  glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
  sc->glist = geomList(sc,mesh);
  sc->type |= S_FOLLOW;

  /* color list */
  setupPalette(sc,mesh);
  sc->stream = NULL;

  initGrafix(sc,mesh);
  return(1);
}

#!BPY

#(C) 2008 Sean Brennan

import Blender
from Blender import Image
from Blender import Window
from Blender import Mesh
from Blender import Scene
from Blender import Texture,Material
from Blender import Modifier
#

def LoadImage(filename):
  image = Image.Load(filename)
  print "Image from", image.getFilename(),
  print "loaded to obj", image.getName()
  MakeMesh(image)

def DefaultImage():
  image = Image.GetCurrent()
  return image

def MakeMesh(image):
  """This takes an image as input and creates a mesh that is an extrusion
  of the mesh treated as a height field, with the zero value areas cut
  from the mesh.
  """

  epsilon = 1.0 / 10000000.0
  set_aspect_ratio = True
  scale_factor = 2.0
  # scale the z value
  zscale = -1.2
  xscale = 14.2
  yscale = 14.2

  xmax, ymax = image.getMaxXY()
  xmin, ymin = image.getMinXY()
  name = image.getName()
  depth = image.depth
  has_data = image.has_data

  xres, yres = image.getSize()
  inv_xres = 1.0 / float(xres)
  inv_yres = 1.0 / float(yres)


  # scale output mesh to have correct aspect ratio 
  # to fit within a unit cube
  if set_aspect_ratio:
    aspect_ratio = float(xres) * inv_yres
    if aspect_ratio > 1.0:
      inv_xres *= 1.0
      inv_yres *= 1.0 / aspect_ratio
    else:
      inv_xres *= aspect_ratio
      inv_yres *= 1.0

  # scale the x and y together
  inv_xres *= scale_factor
  inv_yres *= scale_factor

  print ("xres: %d, yres: %d, xinv %f, yinv %f" % (xres,
	yres, inv_xres, inv_yres))
  print ("depth: %d, has_data: %s, name: %s" % (depth, has_data, name))

  coords = []
  faces = []
  zero_verts = {}

  # Create coordinate array, mark zero z vertices.
  count = 0
  for y in range(yres):
    for x in range(xres):
      r, g, b, a =  image.getPixelF(x, y)
      v = g + r * 256 + b * 256 * 256
      if v > 125:
        v = 125
      coords.append([
		(float(x) * inv_xres - 0.5) * xscale,
		(float(y) * inv_yres - 0.5) * yscale,
		v * zscale])
      #if v < epsilon:
      if v < epsilon and False:
        print "Z: %d" % count,
        zero_verts[count] = 4
      count +=1

  # Create face list.  Decrement zero verts.
  for y in range(yres - 1):
    for x in range(xres - 1):
      p1 = x + (y * xres)
      p2 = p1 + 1  # clockwise?
      p3 = x + (y + 1) * xres + 1
      p4 = p3 - 1 
      if (coords[p1][2] < epsilon and 
          coords[p2][2] < epsilon and
          coords[p3][2] < epsilon and
          coords[p4][2] < epsilon and False):
        zero_verts[p1] -= 1
        zero_verts[p2] -= 1
        zero_verts[p3] -= 1
        zero_verts[p4] -= 1
      else:
        faces.append([p1, p2, p3, p4])

  # Adjust edges for unused zeros
  for y in range(yres):
    p1 = y * xres
    if zero_verts.has_key(p1):
      zero_verts[p1] -= 2
    p1 = p1 + xres - 1
    if zero_verts.has_key(p1):
      zero_verts[p1] -= 2
  for x in range(xres):
    p1 = x
    if zero_verts.has_key(p1):
      zero_verts[p1] -= 2
    p1 = x + xres * (yres - 1)
    if zero_verts.has_key(p1):
      zero_verts[p1] -= 2

  p1 = 0
  if zero_verts.has_key(p1):
    zero_verts[p1] += 1
  p1 = xres -1
  if zero_verts.has_key(p1):
    zero_verts[p1] += 1
  p1 = (yres - 1) * xres
  if zero_verts.has_key(p1):
    zero_verts[p1] += 1
  p1 = p1 + xres - 1
  if zero_verts.has_key(p1):
    zero_verts[p1] += 1

  # Filter vert list and remove unused zeros
  new_verts = []
  remap = {}
  new_count = 0
  for v in range(len(coords)):
    is_zero = zero_verts.has_key(v)
    if not is_zero or zero_verts[v] > 0:
      remap[v] = new_count
      new_verts.append(coords[v])
      new_count +=1
 
  # Re Map old coords to new coords in face list
  new_faces = []
  for f in faces:
    #print "Making face: %s" %  f
    n1 = remap[f[0]] 
    n2 = remap[f[1]] 
    n3 = remap[f[2]] 
    n4 = remap[f[3]] 
    new_faces.append([n1, n2, n3, n4])

  # Verbatim sample code from Blender.Mesh.__doc__
  editmode = Window.EditMode()    # are we in edit mode?  If so ...
  if editmode: Window.EditMode(0) # leave edit mode before getting the mesh
  me = Mesh.New('myMesh')
  me.verts.extend(new_verts)          # add vertices to mesh
  me.faces.extend(new_faces)           # add faces to the mesh (also adds edges)
  #me.mesh.MFace.smooth(1)

  scn = Scene.GetCurrent()          # link object to current scene
  ob = scn.objects.new(me, 'myObj')

  if editmode: Window.EditMode(1)  # optional, just being nice
  # End Verbatim code
  return ob
 
def setTex(ob):
  scn = Scene.GetCurrent()          # link object to current scene
  #ob = scn.objects.get('myObj')
  mat = Material.New('myMat')
  mat.rgbCol = [0.9, 0.9, 0.9]

  footex = Texture.New('footex')
  footex.setType('Image')
  img = Image.Load('/tmp/t-XXXXXXXX.tga')
  footex.image = img
  #mat.tex.type = Texture.Types.IMAGE

  mat.setTexture(0, footex)
  for potential_ob in scn.objects:
   if potential_ob.name == 'myObj':
     ob = potential_ob
     break
  ob.setMaterials([mat])
  ob.colbits = 1
  print 'I did what you asked me to!'
  for ob in scn.objects: print ob.name

def setSubsurf():
  scn = Scene.GetCurrent()          # link object to current scene
  for potential_ob in scn.objects:
   if potential_ob.name == 'myObj':
     ob = potential_ob
     break

  mods = ob.modifiers
  mod = mods.append(Modifier.Types.SUBSURF)
  mod[Modifier.Settings.LEVELS] = 1
  mod[Modifier.Settings.RENDLEVELS] = 2

def fs_callback(filename):
  ob = LoadImage(filename)

ob = LoadImage('/tmp/r-XXXXXXXX.tga')
setTex(ob)
#setSubsurf()
#Blender.Window.FileSelector(fs_callback, "Import Texture Mesh")
#also    	ImageSelector(callback, title, filename) 

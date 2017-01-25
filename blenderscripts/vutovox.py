"""Read the voxel data and get max_min_."""
import numpy as np
import scipy.misc

infile = open('voxel3.csv', 'r')
max_x, max_y, max_z, max_i = -np.inf, -np.inf, -np.inf, -np.inf
min_x, min_y, min_z, min_i = np.inf, np.inf, np.inf, np.inf

rawvox = []

for line in infile:
    (x, y, z, i) = [int(x) for x in line.split(",")]
    rawvox.append([x, y, z, i])

    if x > max_x:
        max_x = x
    if x < min_x:
        min_x = x
    if y > max_y:
        max_y = y
    if y < min_y:
        min_y = y
    if z > max_z:
        max_z = z
    if z < min_z:
        min_z = z
    if i > max_i:
        max_i = i
    if i < min_i:
        min_i = i

diff_x = max_x - min_x
diff_y = max_y - min_y
diff_z = max_z - min_z
diff_i = max_i - min_i

# X, Y, Z, dimensions (resolution) of the voxel datacube.
nx, ny, nz = diff_x+1, diff_y+1, diff_z+1
header = np.array([nx, ny, nz, 1])

print "No of elems", len(rawvox)
print 'max_ x:', max_x, 'y:', max_y, 'z:', max_z, 'i', max_i
print 'min_ x:', min_x, 'y:', min_y, 'z:', min_z, 'i', min_i
print "diffx:", diff_x, 'y', diff_y, 'z', diff_z, 'i', diff_i

sparsevox = []
for row in rawvox:
    x = row[0] - min_x
    y = row[1] - min_y
    z = row[2] - min_z
    #normalize the intensity to 0..1
    i = (float(row[3])-min_i)/max_i
    sparsevox.append([x,y,z,i])

densevox = np.zeros((nx, ny, nz))

for elem in sparsevox:
    x, y, z, i = elem[0], elem[1], elem[2], elem[3]
    #print(x,y,z,i)
    densevox[x][y][z] = i
print densevox.shape

layercnt = 0

for layer in densevox:
    layercnt += 1
    filename = "layer%04d.jpg" % layercnt
    print "writing file", filename, layer.shape

    scipy.misc.toimage(layer, cmin=0.0, cmax=1.0).save(filename)
print "now writing dense format"
denseflat= np.transpose(densevox).flatten()

#binfile = open('city.bvox','wb')
#header.astype('<i4').tofile(binfile)
#denseflat.astype('<f4').tofile(binfile)
#binfile.close()

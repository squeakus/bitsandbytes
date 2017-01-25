# run with command pvbatch --use-offscreen-rendering batchimage.py 

import sys
from paraview.simple import *
# Load the state


if len(sys.argv) < 2: 
    print "Error: please specify filename"
    exit()

print "saving image", sys.argv[1]

paraview.simple._DisableFirstRenderCameraReset()


servermanager.LoadState("glyphs.pvsm")
view = GetRenderView()
SetActiveView(view)

#GetDisplayProperties().CubeAxesVisibility = 0

Render()
WriteImage(sys.argv[1]+".png")


# NCS Example for converting tensorflow network to blob

requires NCSDK2
### ncstrain.py 
trains the network on mnist dataset

### ncsinference.py
Removes the learning layers, unnecessary placeholders and names input and output layers.

to convert to blob:
`mvNCCompile mnist_inference.meta -s 12 -in input -on output -o mnist_inference.graph`



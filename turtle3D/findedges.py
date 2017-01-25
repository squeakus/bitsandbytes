from PIL import Image, ImageFilter

image = Image.open('anchor.png')
image = image.filter(ImageFilter.FIND_EDGES)
image.save('edges.png')

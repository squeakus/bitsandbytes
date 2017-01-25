#include <stdio.h>
#include <tiffio.h>


int main ()
{
  TIFF *tif=TIFFOpen("sample.tif", "r");
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); 
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); 
  
  uint32 npixels=width*h;
  raster=(uint32 *) _TIFFmalloc(npixels *sizeof(uint32));
  TIFFReadRGBAImage(tif, width, height, raster, 0);
  TIFFClose(tif);
  return 0;
}

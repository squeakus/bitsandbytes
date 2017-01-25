#include "tiffio.h"

int main ()
{
  TIFF *out= TIFFOpen("new.tif", "w");
  return 0;
}

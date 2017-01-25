#ifndef HAS_TIFFMAT

#define HAS_TIFFMAT

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tiffio.h>

using namespace cv;

/** Static class containing methods for reading and writing tiff images. 
  * Also 16 Bit images are supported.
  *
  * 2011, Konrad Wenzel, konwen@gmx.de
  */


class Tiffmat
{
public:
	
	/** Reads images from TIFF files and returns cv::Mat.
	/* An array of type cv::Mat is returned with the same number of channels
	/* stored in the file - e.g. type CV_8UC3. 16 Bit images are 
	/* also supported and returned as Mat of type CV_16UCn, where n is the 
	/* number of channels. Only non-compressed images or LZW compressed images
	/* can be read. Pass the flag "0" to enforce a returned single-channel
	/* grayscale image.
	/* \return image
	*/
	static cv::Mat tiffread( const char * filename,	/**< path / filename */
		                     int flags = 1		/**< flags (0 forces grayscale image) */
							 )
	{
		Mat dst;	// Destination matrix

		try
		{
			TIFF * tif;
			uint16 spp, bpp, photo, compression;
			uint32 imageWidth, imageHeight;
			uint32 tileWidth = 0, tileHeight = 0;		

			// Suppress error messages from libtiff
			TIFFSetErrorHandler(0);
			TIFFSetWarningHandler(0);
			
			// Read tiff
			tif = TIFFOpen(filename, "r");
			if (!tif)
			{
				//fprintf (stderr, "Can't open %s for reading\n", filename);
				return(dst);
			}
			else
			{
				// Read header
				TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
				TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageHeight);
				TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
				TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);
				TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp);
				TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
				TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
				TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression);

				// Get CV Mat type
				int m_type = -1;
				if(bpp==8  && spp==1) m_type = CV_8U;
				if(bpp==8  && spp==2) m_type = CV_8UC2;
				if(bpp==8  && spp==3) m_type = CV_8UC3;
				if(bpp==8  && spp==4) m_type = CV_8UC4;
				if(bpp==16 && spp==1) m_type = CV_16U;
				if(bpp==16 && spp==2) m_type = CV_16UC2;
				if(bpp==16 && spp==3) m_type = CV_16UC3;
				if(bpp==16 && spp==4) m_type = CV_16UC4;
				if(bpp==32 && spp==1) m_type = CV_32FC1;
				if(bpp==32 && spp==2) m_type = CV_32FC2;
				if(bpp==32 && spp==3) m_type = CV_32FC3;
				if(bpp==32 && spp==4) m_type = CV_32FC4;
				if(bpp==64 && spp==1) m_type = CV_64FC1;
				if(bpp==64 && spp==2) m_type = CV_64FC2;
				if(bpp==64 && spp==3) m_type = CV_64FC3;
				if(bpp==64 && spp==4) m_type = CV_64FC4;

				// Read only if no compression (1) or LZW compression were applied
				if(compression==1 || compression==5)
				{

					// Read in tiled TIFF
					if(tileWidth>0&&tileHeight>0)
					{
						char * buf = new char[TIFFTileSize(tif)]; 
						dst.create(imageHeight,imageWidth,m_type);
						dst.setTo(0);
						for (uint32 y = 0; y < imageHeight; y += tileHeight)
						{
							for (uint32 x = 0; x< imageWidth; x += tileWidth)
							{											
								// Read tile and make matrix
								TIFFReadTile ( tif, buf, x, y, 0, 0 );
								Mat tile_data( tileHeight, tileWidth, m_type, buf);			
								
								// Irregular Tile size at the image borders
								uint32 current_tileWidth =  ( (imageWidth  - x) < tileWidth ) ? imageWidth  - x : tileWidth;
								uint32 current_tileHeight = ( (imageHeight - y) < tileHeight) ? imageHeight - y : tileHeight;

								// Determine destination roi and copy source roi data
								Mat dst_roi = dst ( Range(y,y+current_tileHeight), Range(x,x+current_tileWidth) );
								tile_data ( Range(0,current_tileHeight), Range(0,current_tileWidth)).copyTo(dst_roi);	
							}
						}		
						delete [] buf;
					}
					else	// Scanline TIFF
					{
						size_t linesize = TIFFScanlineSize(tif);
						dst.create(imageHeight,imageWidth,m_type);
						for (uint32 i = 0; i < imageHeight; i++)
							TIFFReadScanline(tif, &dst.data[i*linesize], i, 0);
					}										
					
					// Alternative: ReadRGBA method from libtiff - requires (too) much memory
					
					//else	
					//{
					//	dst.create(imageHeight,imageWidth,CV_8UC4);
					//	TIFFReadRGBAImage(tif, imageWidth, imageHeight, (uint32*)dst.data, 0);		
					//	cv::flip(dst,dst,0);
					//	bpp = 8; spp = 4;	// method always returns 32 Bit image (4 channels) 
					//}


					// Swap channels for color image
					if(photo==2 && spp==3)	cvtColor(dst, dst, CV_BGR2RGB);
					if(photo==2 && spp==4)	cvtColor(dst, dst, CV_BGRA2RGBA);

					// Convert to greyscale image if flag "0" was specified	and color image was read
					if(flags==0 && photo==2)	
					{
						if(spp==3||spp==4) 
						{
							Mat dst_gray;
							if(spp==3)	cvtColor(dst, dst_gray, CV_RGB2GRAY);
							if(spp==4)	cvtColor(dst, dst_gray, CV_RGBA2GRAY);
							dst.release();
							dst = dst_gray;
						}
						else
						{
							//printf (stderr, "Couldn't find a conversion to greyscale\n" );
							dst.release();	
						}
					}			

					TIFFClose(tif);
					return(dst);
				}
				else // Print error message if compression is not supported, eg. compresion 7 (JPEG)
				{
					//fprintf (stderr, "Error while reading image.\n");
					//fprintf (stderr, "Unsupported compression type in TIF\n");
					return(dst);
				}			
			}
		}
		catch(...)
		{
			//fprintf (stderr, "Error while reading image.\n");
			return(dst);
		}

	};

	//* Writes images of type cv::Mat to TIFF file.
	/* The written image has the same depth and number of channels
	/* as the source matrix, which is passed as reference of type
	/* cv::Mat.
	*/
	static void tiffwrite( const char* filename,	/**< destination path / filename */
			               Mat& src			/**< reference to source matrix */
						   )
	{

		TIFF * tif = TIFFOpen(filename,"w");
		uint16 spp = 0, bpp = 0, compression = 1;
		uint32 imageWidth = src.cols;
		uint32 imageHeight = src.rows;
		uint32 tileWidth = 0, tileHeight = 0;	
		uint16 photometric = 0;
		
		int mt = src.type();
		if ( mt == CV_8U )		{ bpp = 8;	spp = 1; }
		if ( mt == CV_8UC2 )	{ bpp = 8;	spp = 2; }
		if ( mt == CV_8UC3 )	{ bpp = 8;	spp = 3; }
		if ( mt == CV_8UC4 )	{ bpp = 8;	spp = 4; }
		if ( mt == CV_16U )		{ bpp = 16;	spp = 1; }
		if ( mt == CV_16UC2 )	{ bpp = 16; spp = 2; }
		if ( mt == CV_16UC3 )	{ bpp = 16; spp = 3; }
		if ( mt == CV_16UC4 )	{ bpp = 16; spp = 4; }
		if ( mt == CV_32FC1 )	{ bpp = 32; spp = 1; }
		if ( mt == CV_32FC2 )	{ bpp = 32; spp = 2; }
		if ( mt == CV_32FC3 )	{ bpp = 32; spp = 3; }
		if ( mt == CV_32FC4 )	{ bpp = 32; spp = 4; }
		if ( mt == CV_64FC1 )	{ bpp = 64; spp = 1; }
		if ( mt == CV_64FC2 )	{ bpp = 64; spp = 2; }
		if ( mt == CV_64FC3 )	{ bpp = 64; spp = 3; }
		if ( mt == CV_64FC4 )	{ bpp = 64; spp = 4; }
		
		if( spp == 1 )
			photometric = 1;		// Grayscale, Min is Black
		else
		{
			if( spp == 3 && bpp == 8  )
				photometric = 2;	// Assuming RGB
			else 
				photometric = 5;	// Seperated
		}
		
		
		if (spp == 0 || bpp == 0)
		{
			fprintf ( stderr, "Image %s could not be written.\n", filename );
			fprintf ( stderr, "Matrix depth not supported.\n" );
		}
		else
		{
			// Read tiff
			tif = TIFFOpen(filename, "w");
			if ( !tif )
			{
				fprintf (stderr, "Can't open %s for writing\n", filename);
			}
			else
			{
				// Write header
				TIFFSetField (tif, TIFFTAG_IMAGEWIDTH, imageWidth );  
				TIFFSetField (tif, TIFFTAG_IMAGELENGTH, imageHeight );   
				TIFFSetField (tif, TIFFTAG_SAMPLESPERPIXEL, spp );  
				TIFFSetField (tif, TIFFTAG_BITSPERSAMPLE, bpp );    
				TIFFSetField (tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    
				TIFFSetField (tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
				TIFFSetField (tif, TIFFTAG_PHOTOMETRIC, photometric);

				// Linesize and buffer
				tsize_t linesize = TIFFScanlineSize ( tif );
				unsigned char * buf = new unsigned char[linesize];
				
				// Set stripsize
				TIFFSetField( tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize( tif, imageWidth*spp ));
				
				// Swap channels for color image
				Mat src_r = src;
				if( photometric==2 && spp==3)	cvtColor(src, src_r, CV_RGB2BGR);
				if( photometric==2 && spp==4)	cvtColor(src, src_r, CV_RGBA2BGRA);
				
				// Write data to image
				for( int32 row = 0; row < src.rows; row++)
				{
					unsigned char * ptr = src.data + (src_r.step * row);
					memcpy ( buf, ptr, linesize );
					if ( TIFFWriteScanline ( tif, buf, row, 0 ) < 0 )
						break;
				}
				
				delete [] buf;
				TIFFClose( tif );
			}
		}


	};


	// Read in Image using tiffreader or cv reader when no tiff
	static Mat readImage( const char* img_path, bool force_grayscale = false )
	{
		Mat dst;

		// Read in grayscale image
		int grayflag = ( force_grayscale == false ) ? 1 : 0;

		// Reading images
		printf("Reading %s ... ", img_path);	

		// Try tiffreader first, since cv::imread doesn't provide 16 Bit support
		dst = Tiffmat::tiffread ( img_path, grayflag );
		if( dst.empty() == true ) 
			dst = imread ( img_path, grayflag ); 
			
		if( dst.empty() == false) 
		{
			printf("Done. [ %i x %i (%iMP) ]",
				dst.cols, dst.rows, cvRound ( (double) dst.cols * dst.rows / 1000000 ));
			if( dst.depth() == 2 ) 
				printf(" (16 Bit)\n"); 
			else if ( dst.depth() == 0 )
				printf(" (8 Bit)\n");
			printf("\n");
		}
		else
		{
			fprintf( stderr, "Failed. \n\nCheck path, file extension and image format.\n\n", img_path);
			fprintf( stderr, "Supported file formats:\n");
			fprintf( stderr, "png, tiff, tif, bmp, dip, jpg, jpeg, jpe, jp2, sr, ras, pbm, pgm, ppm\n\n");
			system("Pause");
			exit(0);
		}

		return ( dst );
	
	};

	

}; // end of class


#endif

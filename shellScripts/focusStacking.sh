#!/bin/bash
#focusStacking.sh
#Author: José Juan Escudero <code@ramso.net>
#Web: http://www.ramso.net
#creation date: 25/05/2014


SELF=`basename $0`	# Ouselve
DIR=""
ALIGN=0			# Not Align the images by default
QUIET=0			# not too quiet by default

EXPOSURE="0"		# Default exposure compensation
SATURATION="0"	# Default saturation
CONTRAST="1"		# Default Contrast
CONTRASTWS=""	#Default value for Contrast Window Size
CONTRASTES="" 	#Value for Contrast edge scale
GRAY=""		# Value for Gray proyector
OUTSUF="out"		# Default output directory
WORSUF="tmp"		# Default working directory
FSFILE="focusStacking.jpg" # Result sile name
PARAMS=""
DCRAWP="-4 -T -w -p embed"
displayHelp() {
	echo "Focust Stacking of a serie of images."
	echo "Based on the articles of Edu Pérez - http://fotoblog.edu-perez.com"
	echo "and Pat Daviv - http://blog.patdavid.net/"
	echo
	echo "Usage: $SELF [OPTION] DIR"
	echo -e "  -a\t\tAlign images"
	echo -e "  -q\t\tQuiet"
	echo -e "  -h\t\tThis help"
	echo -e "  -w\t\tContrast Window Size"
	echo -e "  -e\t\tContrast Edge Scale"
	echo -e "  -g\t\tGray proyector"
	echo -e "  -d\t\tdcraw parameters quoted the text (By default -4 -T -w -p embed) "
	echo
	echo "Report bugs to <code@ramso.net>"
} 

while getopts aqhg:d:e:w: argument
do
        case $argument in
		a)ALIGN=1;;
                q)QUIET=1;;
                h)displayHelp;exit;;
                g)GRAY=$OPTARG;;
                w)CONTRASTWS=$OPTARG;;
                e)CONTRASTES=$OPTARG;;
                d)DCRAWP=$OPTARG;;
        esac
done

shift $(($OPTIND-1))
DIR=$1

if [ -z $DIR ]; then
	displayHelp
	exit
fi
if [ ! -d "$DIR" ]; then
	echo "$DIR is not a valid directory"
	displayHelp
	exit
fi
DIR=$(cd "$DIR" && pwd) #transform to absolute path
# Check of the directory contains only one type of files
if [ `find "$DIR" -maxdepth 1 -type f -exec basename {} \; | sed "s/.*\.//g" | tr '[:lower:]' '[:upper:]' | sort -u | wc -l` != 1 ]; then
	echo "Error: Directory contains multiple filetypes"
	exit
fi

#create needed directory
OUTDIR="$DIR"/$OUTSUF
mkdir -p "$OUTDIR"
WORKDIR="$DIR"/$WORSUF
mkdir -p "$WORKDIR"

FILES=(`find "$DIR" -maxdepth 1 -type f -print | sort`) # List files in the selected directory
filetype=`basename "${FILES[0]}" | sed "s/.*\.//g" | tr '[:lower:]' '[:upper:]'` # Get file extension
#if [ $filetype = "JPG" ] || [ $filetype = "CR2" ] || [ $filetype = "NEF" ]; then
if [ $filetype = "JPG" ] || [ -z "`dcraw -i ${FILE[0]} | grep Cannot`" ]; then
	if [ $filetype != "JPG" ]; then
		if [ $QUIET = 0 ]; then
			echo "Devloping RAW files"
		fi
		
		if [ $QUIET = 0 ]; then
		  dcraw -v $DCRAWP ${FILES[*]}
		else
		  dcraw $DCRAWP ${FILES[*]}
		fi
                echo "WORK:$WORKDIR"
                echo "PHOTO:$DIR"
		mv $DIR/*.tiff "$WORKDIR"
		FILES=("$WORKDIR"/*.tiff)
	fi
	if [ $ALIGN = 1 ]; then
		if [ $QUIET = 0 ]; then
			echo "Aligning images"
			align_image_stack -a "$WORKDIR"/AIS_ ${FILES[*]} 
		else
			align_image_stack -a "$WORKDIR"/AIS_ ${FILES[*]} >/dev/null 2>&1
		fi
		FILES=("$WORKDIR"/AIS_*.tif)
	fi
	if [ ! -z "$GRAY" ]; then
		PARAMS=$PARAMS"--gray-projector="$GRAY" "
	fi
	if [ ! -z "$CONTRASTES" ]; then
		PARAMS=$PARAMS"--contrast-edge-scale="$CONTRASTES" "
	fi
	if [ ! -z "$CONTRASTWS" ]; then
		PARAMS=$PARAMS"--contrast-window-size="$CONTRASTWS" "
	fi
	
	if [ $QUIET = 0 ]; then
		echo "Generating Enfused image"
		enfuse -v -o "$OUTDIR"/focusStacking.tif --exposure-weight=$EXPOSURE --saturation-weight=$SATURATION --contrast-weight=$CONTRAST --hard-mask $PARAMS ${FILES[*]} 
	else
		enfuse -o "$OUTDIR"/focusStacking.tif --exposure-weight=$EXPOSURE --saturation-weight=$SATURATION --contrast-weight=$CONTRAST --hard-mask $PARAMS ${FILES[*]} >/dev/null 2>&1
	fi
	echo "Ending process"
else
	echo "Unsupported file type: $filetype"
fi

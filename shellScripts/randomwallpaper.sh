#!/bin/bash

WPDIR="$HOME/Pictures/wallpaper"

random=true
apply=true
wpfile=""

function usage {
	if [ $1 -eq 1 ]; then
		stream=2
		exitcode=255
	else
		stream=1
		exitcode=0
	fi
	echo "Usage: $(basename $0) [-n|--noapply] [-h|--help] [wallpaper_location]" >&$stream
	echo "If wallpaper location is not given a random wallpaper from $WPDIR will be chosen" >&$stream
	exit $exitcode
}

# handle arguments
while [ $# -gt 0 ]; do
	if [ "$1" = "--help" -o "$1" == "-h" ]; then
		usage 0
	elif [ "$1" = "--noapply" -o "$1" = "-n" ]; then
		apply=false
	else
		if ! $random; then
			usage 1
		elif [ ! -f "$1" ]; then
			echo "file '$1' not found" >&2
			exit 1
		fi
		random=false
		{ cd $(dirname "$1"); dir=$(pwd); }
		wpfile="$dir/$(basename "$1")"
	fi
	shift
done

if $random; then
	wpfile1=$(ls "$WPDIR"/*.jpg | sort -R | head -n 1)
	echo "chose $wpfile1" >&2
fi

if $random; then
	wpfile2=$(ls "$WPDIR"/*.jpg | sort -R | head -n 1)
	echo "chose $wpfile2" >&2
fi

cat >$HOME/.config/nitrogen/bg-saved.cfg <<EOF
[xin_0]
file=$wpfile1
mode=1
bgcolor=#000000

[xin_1]
file=$wpfile2
mode=1
bgcolor=#000000
EOF
/bin/sleep 2s
nitrogen --restore

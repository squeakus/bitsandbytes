#!/bin/bash

# Usage: backupFilesInCurrentVersion.sh <fileName> <backup dir>

# Given a list of files in the form:
# <commit sha> <size in bytes> <file name (full path from repo root)>
# List the files that exist in the current checked-out point in the history and
# copy them to <backup dir>

mkdir "$2"

while read sha size name; do
	if [ -f "$name" ]; then
		echo "$name"
		cp "$name" "$2"
	fi
done < "$1"

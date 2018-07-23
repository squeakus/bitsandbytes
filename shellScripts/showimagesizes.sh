#!/bin/bash

for img in *.jpg; do
    echo $img
    convert $img -print "Size: %wx%h\n" /dev/null
done

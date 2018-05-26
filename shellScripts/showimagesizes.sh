#!/bin/bash

for img in *.jpg; do
    convert $img -print "Size: %wx%h\n" /dev/null
done

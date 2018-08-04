#!/bin/bash
rm -f quick.jpg
raspistill -o quick.jpg
xdg-open quick.jpg

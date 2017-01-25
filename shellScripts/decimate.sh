#!/bin/bash

for img in *.laz; do
  echo "Thinning once"
  echo "las2las $img --output thinned$img -t 2"
done

for img in thinned*.laz; do
  echo "Thinning twice"
  echo "las2las $img --output double$img -t 2"
done

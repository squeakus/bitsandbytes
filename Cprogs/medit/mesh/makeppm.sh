#!/bin/bash
for file in *.mesh; do
    cmd='ffmedit -xv 200 200 '$file
    $cmd
done

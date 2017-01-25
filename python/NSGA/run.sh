#!/bin/bash
`rm -rf *.dat`
`python Problem.py`
`gnuplot front.plot`
`gnome-open front.ps`

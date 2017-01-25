#!/bin/bash
`R CMD BATCH plotImages.r`
`convert plot*.ps front.gif`

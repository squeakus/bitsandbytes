#!/bin/bash
var="hello how are you. im fine"
#length
echo ${#var}
#strip
echo ${var:(-4)}
#slice
echo ${var:7}
echo ${var:6:3}
echo ${var#.*}
#index
echo `expr index "$var" "l"`
#substring delete front
echo ${var#*are}
echo ${var#*a}
echo ${var#*[al]}
#substring delete back
echo ${var##*are}
echo ${var##*a}
echo ${var##*[al]}


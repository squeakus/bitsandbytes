#!/bin/bash

unset MV_TOOLS_DIR
unset MDK_INSTALL_DIR
unset MV_COMMON_BASE

if [ $# -lt 1 ];
then
    echo "please specify mdk version: switchmdk.sh <rx><r9><dev>"
    exit 0
fi
echo "setting path to /home/jonathan/code/mdk-$1"
export MV_TOOLS_DIR=/home/jonathan/code/mdk-$1/tools
export MDK_INSTALL_DIR=/home/jonathan/code/mdk-$1/mdk/common
export MV_COMMON_BASE=/home/jonathan/code/mdk-$1/mdk/common

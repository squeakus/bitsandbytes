#!/bin/bash
T="$(date +%s)"

sleep 2

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
printf "Pretty format: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"


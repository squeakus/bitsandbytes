#!/bin/bash

# Totally remove $1 from the repo history, where $1 is an object ID. This
# changes the project history.

git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch $1" --prune-empty --tag-name-filter cat -- --all

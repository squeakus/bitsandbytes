#!/bin/bash
ssh michaelfenton@galapagos.ucd.ie zip -r ./truss1.zip ./truss1/CompletedRuns
scp michaelfenton@galapagos.ucd.ie:truss1.zip .
ssh michaelfenton@galapagos.ucd.ie rm truss1.zip

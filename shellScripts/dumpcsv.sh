#!/usr/bin/env bash

# obtains all data tables from database
TS=`/C/Users/byrnejon/code/sqlite-tools/sqlite3.exe $1 "SELECT tbl_name FROM sqlite_master WHERE type='table' and tbl_name not like 'sqlite_%';"`

# exports each table to csv
for T in $TS; do

/C/Users/byrnejon/code/sqlite-tools/sqlite3.exe $1 <<!
.headers on
.mode csv
.output ${T}
select * from $T;
!

done

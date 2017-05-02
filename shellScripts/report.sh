#!/bin/bash
LOGDATE=$(date +%Y%m%d)

oldmooocpp=`tail -n 1 prevresult.txt | awk '{print $1}'`
mooocpp=`cat MvBotCppResults.txt | wc | awk '{print $1}'`
mooopy=`tail -2 MvBotPythonResults.txt `
zipfile=results$LOGDATE.zip

zip $zipfile Mv*.txt

echo "" > report.txt
echo "moooo score:" > report.txt
echo "C/CPP Errors: $mooocpp previous score: $oldmooocpp" >> report.txt
echo "PYTHON lint score: $mooopy" >> report.txt

echo "$mooocpp $mveotcpp $mvolacpp" >> prevresult.txt

uuencode $zipfile $zipfile | cat report.txt - | mailx -s "Code report for $LOGDATE" mooooo.mooo@blah.com

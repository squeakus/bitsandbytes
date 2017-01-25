#!/bin/bash
COUNTER=5
`exiv2 blah.jpg > /dev/null`
RETVAL=$?
[ $RETVAL -eq 0 ] && echo saved
[ $RETVAL -ne 0 ] && `echo "Image $COUNTER has failed to save"|
                      mailx -s "Image capture failed" jonathanbyrn@gmail.com`

echo script finished


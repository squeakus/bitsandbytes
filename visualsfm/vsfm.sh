#!/bin/bash

rm *.log 2> /dev/null
starttime=$(date +%s)
echo "starting at" `date`
workfolder="`pwd`/"
echo "starting at" `date` >> out.log

#SIFT and low density reconstruction
#~/vsfm/bin/VisualSFM sfm $workfolder model.nvm
~/Applications/vsfm/bin/VisualSFM sfm+pairs $workfolder model.nvm @100
echo "finished matching pairs at" `date` >> out.log
#Higher density sparse reconstruction
~/Applications/vsfm/bin/VisualSFM sfm+loadnvm+pmvs model.nvm model.nvm
#Dense reconstruction
#~/vsfm/bin/VisualSFM sfm+loadnvm+cmvs model.nvm model.nvm

echo "finished dense reconstruction at" `date`
echo "finished dense reconstruction at" `date` >> out.log
endtime=$(date +%s)
diff=$(($endtime-$starttime))
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed." >> out.log


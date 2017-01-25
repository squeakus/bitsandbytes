#/bin/bash
echo 'date'":starting experiment Bark"> log.txt
python comparedetectors.py Bark/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Bark/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Bark/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Bark/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Bark/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Boat"> log.txt
python comparedetectors.py Boat/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Boat/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Boat/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Boat/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Boat/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Graf"> log.txt
python comparedetectors.py Graf/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Graf/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Graf/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Graf/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Graf/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Trees"> log.txt
python comparedetectors.py Trees/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Trees/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Trees/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Trees/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Trees/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Wall"> log.txt
python comparedetectors.py Wall/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Wall/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Wall/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Wall/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Wall/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Bikes"> log.txt
python comparedetectors.py Bikes/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Bikes/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Bikes/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Bikes/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Bikes/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment Leuven"> log.txt
python comparedetectors.py Leuven/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py Leuven/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py Leuven/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py Leuven/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py Leuven/ brisk
echo 'date'":brisk finished">> log.txt
echo 'date'":starting experiment UBC"> log.txt
python comparedetectors.py UBC/ sift
echo 'date'":sift finished">> log.txt
python comparedetectors.py UBC/ surf
echo 'date'":surf finished">> log.txt
python comparedetectors.py UBC/ orb
echo 'date'":orb finished">> log.txt
python comparedetectors.py UBC/ akaze
echo 'date'":akaze finished">> log.txt
python comparedetectors.py UBC/ brisk
echo 'date'":brisk finished">> log.txt

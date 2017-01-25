ROOT_FOLDER="/home/jonathan/Jonathan/programs/shellScripts"
EMAIL="jonathanbyrn@gmail.com"

function check_partition {
    echo "checking $1"
# grep/awk the percentage used
    SPACE=`df -h | grep $1 | awk '{print $5}'`
    SPACE=${SPACE%'%'}

#if no previous value exists set it to zero
    PREV_FILE=$ROOT_FOLDER$1".txt"
    if [ ! -e $PREV_FILE ]; then
        echo "-1" > $PREV_FILE
    fi

# get prev value from the file (rounded to tens)
    PREV_SPACE=`cat $PREV_FILE`
    MOD=$(expr $SPACE % 10)
    let "ROUNDED = $SPACE - $MOD"

# check for increase
    if [ $ROUNDED -gt $PREV_SPACE ]; then
        echo "partition $1 on server $HOSTNAME now $ROUNDED% full"
        echo "$ROUNDED" > $PREV_FILE
    fi

# check if above 90%
    if [ $SPACE -gt 60 ]; then
        echo "[CRITICAL] partition $1 running out of space($SPACE%) on $HOSTNAME"
    fi
}

#create a dev folder to keep track of the storage device sizes
    if [ ! -d $ROOT_FOLDER"/dev" ]; then
        echo "no dev folder"
        mkdir  $ROOT_FOLDER"/dev"
    fi

#list of all partitions and check them
PARTITIONS=`df -h | grep dev/sd | awk '{print $1}'`

for partition in $PARTITIONS
do
    check_partition $partition
done


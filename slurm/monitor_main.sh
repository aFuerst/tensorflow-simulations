#!/bin/bash
timestamp() {
  date +"%s"
}

thread_count=$1
particle_count=$2
folder="./cpu_stats/tf/$particle_count"
rm -rf $folder
cd $folder
log="$thread_count.out"
rm -f $log
start="$(timestamp)"
python ../lj-box/tf/main.py -x -c --parts $particle_count --threads $thread_count &
export PID=$!
fname="../../cpu_stats/tf/$particle_count/top_$thread_count.dat"
rm $fname -f
while [ -n "$(ps cax | grep $PID)" ]; do top -p $PID -bn 1 | tail -n 1 | grep --invert-match "%CPU" | awk -v now=$(date +%s) '{print now,$9}' >> $fname; done

echo $thread_count >> $log
echo $particle_count >> $log
echo $start >> $log
echo $(timestamp) >> $log
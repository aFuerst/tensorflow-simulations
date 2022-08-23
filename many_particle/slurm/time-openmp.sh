#!/bin/bash
timestamp() {
  date +"%s"
}

thread_count=$1
particle_count=$2
num_steps=10000
start="$(timestamp)"
folder="./omp/$particle_count-$thread_count"
log="./times-$num_steps.out"
rm -rf folder
mkdir -p  $folder
cd $folder
echo 0.8442 $particle_count $num_steps $thread_count | ../../../openmp/testing_omp > "./omp-$particle_count-$thread_count" &
export PID=$!
fname="../../cpu_stats/omp/$particle_count/top_$thread_count.dat"
rm $fname -f
touch $fname
while [ -n "$(ps cax | grep $PID)" ]; do top -p $PID -bn 1 | tail -n 1 | grep --invert-match "%CPU" | awk -v now=$(date +%s) '{print now,$9}' >> $fname; done

end="$(timestamp)"
echo $thread_count >> $log
echo $particle_count >> $log
echo $start >> $log
echo $end >> $log
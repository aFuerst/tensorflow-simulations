#!/bin/sh
thread_count=100
while [ $thread_count -ge 5 ]; do
    echo $thread_count
    python ./tf/main.py -x -c --parts 5000 --threads $thread_count &
    export PID=$!
    fname="cpu_stats_5000/top_$thread_count.dat"
    rm $fname -f
    while [ -n "$(ps cax | grep $PID)" ]; do top -p $PID -bn 1 | tail -n 1 | grep --invert-match "%CPU" | awk -v now=$(date +%s) '{print now,$9}' >> $fname; done
    thread_count=$(($thread_count-5))
done

thread_count=1
python ./tf/main.py -x -c --parts 5000 --threads $thread_count &
export PID=$!
fname="cpu_stats_5000/top_$thread_count.dat"
rm $fname -f
while [ -n "$(ps cax | grep $PID)" ]; do top -p $PID -bn 1 | tail -n 1 | grep --invert-match "%CPU" | awk -v now=$(date +%s) '{print now,$9}' >> $fname; done

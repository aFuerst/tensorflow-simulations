#!/bin/sh

min() {
 echo $(($1<48 ? $1 : 48))
}
module load python/3.6.9
pip install --user tensorflow==1.15
rm -rf "./slm_output/tf/*"
rm -rf "./outputs/tf/*"
rm -rf "./cpu_stats/tf/*"
for particle_count in 108 2000 5000
do
  thread_count=1
  mkdir -p "./cpu_stats/tf/$particle_count"
  while [ $thread_count -le 100 ];
  do
    sbatch -n 1 --job-name "af-tf-$particle_count-$thread_count" --cpus-per-task "$(min $(($thread_count+1)))" --output "./slm_output/tf/%x.out" monitor_main.sh $thread_count $particle_count
    thread_count=$(($thread_count+1))
  done
done

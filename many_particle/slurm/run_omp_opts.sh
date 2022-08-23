#!/bin/sh

min() {
 echo $(($1<48 ? $1 : 48))
}

rm -rf "./slm_output/omp/*"
rm -rf "./outputs/omp/*"
rm -rf "./cpu_stats/omp/*"
mkdir -p "./slm_output/omp/"
for particle_count in 108 2000 5000
do
  thread_count=1
  mkdir -p "./cpu_stats/omp/$particle_count"
  while [ $thread_count -le 100 ];
  do
    sbatch -n 1 --job-name "af-omp-$particle_count-$thread_count" --cpus-per-task "$(min $(($thread_count+1)))" --output "./slm_output/omp/%x.out" time-openmp.sh $thread_count $particle_count
    thread_count=$(($thread_count+1))
  done
done

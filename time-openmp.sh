#!/bin/bash

timestamp() {
  date +"%s"
}
cd /extra/alfuerst/sim/
cd ./omp-out/

mkdir ./steps-tims/
cd ./steps-tims/
num_atoms=108
num_steps=10000
while [ $num_steps -le 200000 ]
do
start="$(timestamp)"
mkdir $num_steps
cd $num_steps
echo 0.8442 $num_atoms $num_steps | ./openmp/testing_omp > "./omp-$num_steps"
end="$(timestamp)"
echo $start > "./times-$num_steps"
echo $end >> "./times-$num_steps"
num_steps=$((num_steps+10000))
cd ..
done

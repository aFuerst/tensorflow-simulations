#!/bin/bash

timestamp() {
  date +"%s"
}
cd /extra/alfuerst/sim/

# echo "$(timestamp)"
# rm -rf ./omp-out/
# mkdir ./omp-out/
cd ./omp-out/
# mkdir ./atoms-tims/
# cd ./atoms-tims/
# num_atoms=100
# while [ $num_atoms -le 2000 ]
# do
# start="$(timestamp)"
# mkdir $num_atoms
# cd $num_atoms
# echo 0.8442 $num_atoms 100000 | /home/alfuerst/tensorflow-simulations/lj-box/openmp/testing_omp > "./omp-$num_atoms"
# end="$(timestamp)"
# echo $start > "./times-$num_atoms"
# echo $end >> "./times-$num_atoms"
# num_atoms=$((num_atoms+100))
# cd ..
# done
# cd ..

mkdir ./steps-tims/
cd ./steps-tims/
num_atoms=108
num_steps=10000
while [ $num_steps -le 200000 ]
do
start="$(timestamp)"
mkdir $num_steps
cd $num_steps
echo 0.8442 $num_atoms $num_steps | /home/alfuerst/tensorflow-simulations/lj-box/openmp/testing_omp > "./omp-$num_steps"
end="$(timestamp)"
echo $start > "./times-$num_steps"
echo $end >> "./times-$num_steps"
num_steps=$((num_steps+10000))
cd ..
done

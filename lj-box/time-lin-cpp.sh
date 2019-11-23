#!/bin/bash

timestamp() {
  date +"%s"
}
rm -rf ./lin-cpp-out/
mkdir ./lin-cpp-out/
cd ./lin-cpp-out/
num_atoms=100
while [ $num_atoms -le 2000 ]
do
start="$(timestamp)"
mkdir $num_atoms
cd $num_atoms
echo 0.8442 $num_atoms | /home/alfuerst/tensorflow-simulations/lj-box/cpp/simulate_many_particle_dynamics > "./lin-cpp-$num_atoms"
end="$(timestamp)"
echo $start > "./times-$num_atoms"
echo $end >> "./times-$num_atoms"
num_atoms=$((num_atoms+100))
cd ..
done

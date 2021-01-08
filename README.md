
# Purpose of this readme-
This readme is being wrtten to log the progress made on a specific part of this repository. This part involves developing the surrogate for many particle dynamics.The implementation resides under the folder name "tf". This surrogate will take density as input and predict the energy (kinetic and potential) and temperature of the system.

# Progress
# 12-9
The ordinary verlet works fine and produces the correct result. Result was verified againg the velocity-verlet's TF implementation
The pipeline for implementing Many Particle MD Surrogate using Tensorflow was designed last week and reviewed in this week's meeting. It can found in the repo with the name 'pipeline.png'. There are some fixes suggested by Vikram. I will be including them and updating it this week.


The plan for going about this part is: 
1. Generating the train and validation data in a text file.
2. I will start with coding the TF model in model.py
    - This file will do the training and debugging for the model
3. Integrating the model to the rest of the repo. 


# 12-16
Last week the plan above was further modified as below:

*1. Complete and verfiy the energy calculation functions. Verify energy conservation plot in the existing code first.*                

2. Generating the train and validation data in a text file.

3. I will start with coding the TF model in model.py
    - This file will do the training and debugging for the model

4. Integrating the model to the rest of the repo.

I have started working on verifying the energy part. I was stuck for a long time on finding a way to write the energy tensor to a file. I started with doing it in the same way as positions and forces were being written inside save(). But then I realised that the later were np.arrays. So I tried writing the energy tensor using tf.print and eval() to the std o/p. But I guess because it is a lazy execution, I wasn't able to see the values. I am using TF2.0 and had to disable the eager execution when I started working on this code in the beginning. I am not sure if that's interfering with this.

I have three routes to get to the energy values:
1. Try an eager execution
2. Declare the energy tensors as np.arrays (like positions, forces and velocities) are declared
3. Get the tensor wriiten to a file.

I spent half a day on the 3rd route. Maybe I should explore the other two to get a pace on getting to the energy conservation code.

# 01-06
Status of tasks enlisted above: 
1. The energy calculations were fixed and verified in the current tensorflow code. Everything works similar to C++ code. 
2. Data for surrogate has been generated. The behavioral convergence was identified at 80000 steps. To round off, data was prepared with 100000 steps. All the parameters including input, output and neural were defined for the model. Some of the important ones were: 

##Input Params:
Density from 0.1 to 0.95
##Output Params:
Average Potential Energy per particle
Average Temperature per particle
##Neutral Parameters:
Number of steps = 100000
Equilibrium hit = 5000 steps
Number of particles = 500
Total time = 100s
log_freq = 100

3. The model implementation is in Progress. I have a model with one hidden layer. The is one input and 2 outputs for the Neural network. The hidden layer has 2 neurons for now. I am still working on fine tuning it. The learning rate is set at 0.001 with adam optimizer. I am working on setting the right number of epochs for minimum loss and max accuracy. The data is divided into three sets : training(54 samples), validation(14 samples) and test(18 samples).

4. Integration strategy needs to be discussed with Vikram and Prateek. 

5. I have started looking at the nano code in parallel. 


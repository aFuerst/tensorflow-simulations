
## Purpose of this readme-
This readme is being wrtten to log the progress made on a specific part of this repository. This part involves developing the surrogate for many particle dynamics.The implementation resides under the folder name "tf". This surrogate will take density as input and predict the energy (kinetic and potential) and temperature of the system.

## Progress
12-9
The ordinary verlet works fine and produces the correct result. Result was verified againg the velocity-verlet's TF implementation
The pipeline for implementing Many Particle MD Surrogate using Tensorflow was designed last week and reviewed in this week's meeting. It can found in the repo with the name 'pipeline.png'. There are some fixes suggested by Vikram. I will be including them and updating it this week.


The plan for going about this part is: 
1. Generating the train and validation data in a text file.
2. I will start with coding the TF model in model.py
    - This file will do the training and debugging for the model
3. Integrating the model to the rest of the repo. 


12-16
Last week the plan above was further modified as below:
1. Complete and verfiy the energy calculation functions. Verify energy conservation plot in the existing code first.                //this was added
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

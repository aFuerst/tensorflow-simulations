
## Purpose of this readme-
This readme is being wrtten to log the progress made on a specific part of this repository. This part involves developing the surrogate for many particle dynamics.The implementation resides under the folder name "tf". This surrogate will take density as input and predict the energy (kinetic and potential) and temperature of the system.

## Progress
12-9
The ordinary verlet works fine and produces the correct result. Result was verified againg the velocity-verlet's TF implementation
The pipeline for implementing Many Particle MD Surrogate using Tensorflow was designed last week and reviewed in this week's meeting. It can found in the repo with the name 'pipeline.png'. There are some fixes suggested by Vikram. I will be including them and updating it this week.


The plan for going about this part is: 
1. Generating the train and validation data in a text file.
2. I will start with coding the TF model in model.py
  - this file will do the training and debugging for the model
3. Integrating the model to the rest of the repo. 



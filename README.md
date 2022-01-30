# General Framework for Data Minning
This is a general framework for data minning.  
For better records of experiments, I also use [neptune](https://neptune.ai/) to track results.

# Features
* Modular design. You can implement your model, dataset by defining a new class inherited from *BaseModel* or *BaseDataset*
* Model functions. By given specific configurations, the framework will load, save,and initialize model and schedule learning rates automatically.
* Experiment display. The framework could display settings of the model and experiment results.
* Neptune tracking. For better displays, if enabled, the framework also upload experiment settings, results and any visualization to the neptune platform.
# Things to Note
* This framework is still under development.
* For how to use the framework in detail, please refer to the annotations in code.
# Acknowledgment
The code framework is largely adopted from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) as they provide functions to load, save, and optimize models. Thanks for their greate work!!
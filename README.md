A Learned Shape-Adaptive Subsurface Scattering Model
===================================

Implementation of the paper ["A Learned Shape-Adaptive Subsurface Scattering Model"](https://rgl.epfl.ch/publications/Vicini2019Learned), Siggraph 2019 by Delio Vicini, Vladlen Koltun and Wenzel Jakob.

The implementation is based on [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba).

# Compiling the code 
See the [Mitsuba documentation](http://mitsuba-renderer.org/docs.html). 
The python source code relies on the Mitsuba python bindings, 
so you have to make sure the SCons build system builds them alongside the Mitsuba executable. 
Once Mitsuba is built, run the command `source setpath.sh` to add Mitsuba and the python bindings to the path.

# Dependencies for the training code 
The python code used for training the model was tested using Python 3.6 and requires the following modules: 
* numpy
* tensorflow 1.8
* tqdm
* trimesh
* scikit-image
* matplotlib
* [exrpy](https://github.com/rgl-epfl/exrpy)

The repository also contains an OpenGL viewer to visualize the learned distribution of scattering locations. 
To run this viewer, you have to additionally clone and build a custom version of NanoGUI from https://github.com/dvicini/nanogui. 
The relevant code is on the branch "framebuffer". 

# Using the pre-trained model 
This repository contains the trained model used to generate the results from the paper. 
While this code base also contains all the code for training a new model from scratch, 
it might be more convenient to just run the pre-trained model (e.g. for comparisons)

To do so, first rename the folder `pysrc/outputs_paper` to `pysrc/outputs`. 
The following command then renders the teaser scene (see the files in the `scenes` subfolder):
```
python render_results.py --scenes teaser --spp 32 --techniques 0487_FinalSharedLs7Mixed3_AbsSharedSimComplexMixed3 --ncores 8
```



# Generating the training data 
The model is trained on data set of ground-truth light paths. 
To generate the training data set, the following script has to be run: 

```
python train_scattering_model.py --gentraindata --datasetconfig ScatterData
```

This will take a while (depending on the machine several hours). 
The size of the data set can be decreased for debugging purposes by adjusting `n_scenes` in the `MeshGenerator` class in `datapipeline.py` and `n_train_samples` in the `ScatterData` class in the same file.

Once the data set is generated, it should be written to a subfolder in `pysrc/outputs/vae3d/datasets`. 

# Training the model 
To train the model, the variable `SCATTERDATASIMILARITY` in `pysrc/vae/config.py` has to be set to the training dataset name which should be used. 
By default, the data set generated in the previous step will be called `0000_ScatterData` and nothing has to be changed. 

The model itself can then be trained by running 
```
python train_scattering_model.py --train --config VaeScatter/AbsorptionModel
```


# Rendering using a trained model 

The simplest way to render using a trained model is to use the `render_results.py` script.
We provide the teaser scene of our paper as an example scene in the `scenes` subfolder.
To render it using the trained model, simply run
```
python render_results.py --scenes teaser --spp 32 --techniques 0000_VaeScatter_AbsorptionModel --ncores 8
```
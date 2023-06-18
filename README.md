# CS330 Transfer and Meta-Learning in Graph Neural Networks: A Recommender System Approach

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Screenshots](#screenshots)
* [Status](#status)
<!--* [Inspiration](#inspiration)
* [Contact](#contact)-->

## General info
Code for our CS 330 Deep Multi-Task and Meta Learning Final Project:
Transfer and Meta-Learning in Graph Neural Networks: A Recommender System Approach

Code is divided into two main folders: Meta-Learning and Transfer_Learning 

* Transfer_Learning_and_Joint-Loss contains the code implementation of a GCN rating-prediction model, and scripts used for fine-tuning and joint-loss training experimentation 
* Meta-Learning contains the code base for the MAML and Meta-Graph implementations, the base VGAE link prediction model, and data-processing code 


## Technologies
* python -version 3.8.16
* pytorch -version 1.13.0+cu166
* torch_geometric -version 2.2.0
* numpy
* pandas
* matplotlib

<!--## Baseline Model
keras implementation (https://github.com/divamgupta/image-segmentation-keras/) -->
## Screenshots
<img src="model_selection.png" width="700" />
<img src="ROC_curves_for_LGBM_classifier.png" width="700" />
<img src="feature_importance.png" width="700" />
<!--<img src="PyCaret_feature_importance_best.png" width="500" /> -->



<!--## Setup-->
<!--Available soon-->
<!--Describe how to install / setup your local environement / add link to demo version.-->

<!--## Code Examples
Show examples of usage:
```
from keras_segmentation.models.unet import unet_mini

model = unet_mini(n_classes=4,  input_height=96, input_width=96  )

model.train(
    train_images = "Dataset/train/",
    train_annotations = "Dataset/train_labels/",
    checkpoints_path = "Dataset/checkpoints",
    val_images = "Dataset/test/",
    val_annotations = "Dataset/test_labels/",
    epochs=50, validate=True, batch_size=8, 
    optimizer_name="adam",
    gen_use_multiprocessing=True,
    auto_resume_checkpoint=False,
    val_batch_size=2,
)
```

## Features
List of features ready and TODOs for future development
* Train on 3 different U-NET architecture variants-->

## Results
From feature selection, regularization analysis and the general screening, we can see that model performance is quite similar ( RMSE ∼ 3) and no model offers a very good approximation to BMI yet. Also, in all approaches, leptin, gender and age have come up as the most important features to focus on, as expected and discussed with our mentor.

## Status
Project is: _finished_ <!-- a normal html comment _finished_, _no longer continue_ and why?-->

## Report
CS229 Summer 2022 Paper [link](paper/CS229__BMI_prediction_using_immune_markers.pdf)

<!--## Inspiration-->
<!--Add here credits. Project inspired by..., based on...

<!--## Contact-->
<!--Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me!-->

# coupling_registration_segmentation
This repository is the official repository of the article U-ReSNet: Ultimate coupling of Registration and Segmentation with deep Nets presented in Miccai 2019 (Shenzhen)

The article is available here : https://link.springer.com/chapter/10.1007/978-3-030-32248-9_35

## Train

To train the model, the user may have three different options : 
  - By default, the model will be train only on the registration without having any segmentation task, or deformation of the segmentations masks
  - If the user specify --use-mask, the model will calculate only the registration, but it will apply the deformation on the segmentations masks and use it to calculate a supplementary dice loss
  - If the user specify --segmentation, the model will calculate both the registration and the segmentation as in the article
  

Concerning the deformation part, other options are availabe :
  - --use-affine to add a affine transformation to the non rigid transformation
  - --deform-reg and --affine-reg for the weights regularisation of the non rigid and affine layers
  - --freeze-non-rigid and --freeze-affine to freeze the non rigid and affine layers
  

Three different trained model are available corresponding to the different options --use-mask, --segmentation. The models availabla have been trained with the following command line :
  - ```python3.6 -m coupling_registration_segmentation.main --segmentation --epochs 40 --lr-decrease```
  - ```python3.6 -m coupling_registration_segmentation.main --use-mask --epochs 40 --lr-decrease```

All the other options are the default options.
The train, validation and test split are also available.

## Inference

Concerning the inference, 2 functions are available : 
  - `model_output.py` to predict the output deformation and save and/or plot them.
  - `model_evaluation.py` to predict the output deformation and calculate the dice scores on the different brain structures.

To perform the inference on the two models provided, use the following command line :
  - ```python3.6 -m coupling_registration_segmentation.model_output --pretrained --load-segmentation --all-label --use-mask --aseg --test --plot --load-name model_segmentation```
  - ```python3.6 -m coupling_registration_segmentation.model_output --pretrained --all-label --use-mask --aseg --test --plot --load-name model_use_mask```

Same command line for the `model_evaluation.py` functions


## Data


In this article we use one public dataset : **OASIS 3**.
OASIS 3 was used only for training and testing both segmentation and registration. 
More details are given in the article (about preprocessing and also Freesurfer annotations).

We don't provide the data in this repo.
People can find data on the following link : https://www.oasis-brains.org/ .

To run the code, you need to download the dataset. By default, the data is supposed to be in the folder 
`coupling_registration_segmentation\data\oasis\` .

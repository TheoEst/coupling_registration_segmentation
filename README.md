# coupling_registration_segmentation
This repository is the official repository of the article U-ReSNet: Ultimate coupling of Registration and Segmentation with deep Nets presented in Miccai 2019 (Shenzhen)

The article is available here : https://link.springer.com/chapter/10.1007/978-3-030-32248-9_35

To train the model, the user may have three different options : 
  - By default, the model will be train only on the registration without having any segmentation task, or deformation of the segmentations masks
  - If the user specify --use-mask, the model will calculate only the registration, but it will apply the deformation on the segmentations masks and use it to calculate a supplementary dice loss
  - If the user specify --segmentation, the model will calculate both the registration and the segmentation as in the article
  

Concerning the deformation part, other options are availabe :
  - --use-affine to add a affine transformation to the non rigid transformation
  - --deform-reg and --affine-reg for the weights regularisation of the non rigid and affine layers
  - --freeze-non-rigid and --freeze-affine to freeze the non rigid and affine layers
  

Three different trained model are available corresponding to the different options --use-mask, --segmentation and default. And the train, validation and test split are also available.

Concerning the inference, 2 functions are available : 
  - model_output.py to predict the output deformation and save and/or plot them.
  - model_evaluation.py to predict the output deformation and calculate the dice scores on the different brain structures.

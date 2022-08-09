# Recommended dataset format (Pascal VOC)

Required - two datasets - with rendered images of the equipment and the real images (in separate folders).

Preferred dataset format for this task is Pascal VOC format, originally used here [1](https://paperswithcode.com/dataset/pascal-voc/). The absolute minimum is 100 images per model. 
The recommended amount is 1000. 
The amount of real images can be smaller. 
In my tests I have experimented on the model with 50k rendered images and 10k real images of 30 classes, so 5 to 1. 

Each dataset should be split into 2 parts - trainval (for training the model) and test(for evaluation). Recommended is 85/15% split. 
The files trainval.txt and test.txt under /ImageSets/Main directory contain the list of filenames that belong to the given split. 
The images should be stored in /JPEGImages (for example in .jpg, .jpeg or .png format - note: not with capital letters). 
The annotations for the corresponding image should be stored in /Annotations.

The annotation can be generated for example using [2](https://github.com/heartexlabs/labelImg/). For the rendered images, all objects should be labeled. 
For the real images, although the labels are not required, the .xml anotation files are still needed to describe the image width etc. 
Additionally, the test set should also be labeled for evaluation.
In long term, this will be fixed.
Would be also nice to record the existing models in classes.txt file, if possible. 
Additionally, it would be nice to have the 3D model file, for example in .stp format.


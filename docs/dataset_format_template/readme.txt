Required - two datasets - with rendered images of the equipment and the real images (in separate folders).

Preferred dataset format for this task is Pascal VOC format, originally used here [1]. The absolute minimum is 100 images per model. The recommended amount is 1000. 
The amount of real images can be smaller. In my tests I have experimented on the model with 50k rendered images and 10k real images of 30 classes, so 5 to 1. 

Each dataset should be split into 2 parts - trainval (for training the model) and test(for evaluation). Recommended is 80/20% split. The files trainval.txt and test.txt under /ImageSets/Main directory contain the list of filenames that belong to the given split. 
The images should be stored in /JPEGImages (for example in .jpg or .png format). The annotations for the corresponding image should be stored in /Annotations.
In long term, we only need the annotations for the rendered images, but at the initial stage it might be good to have them for the dataset with real images too. 

Would be also nice to record the existing models in classes.txt file, if possible. 

----------------------------------------------------------------------------------------------------
[1] https://paperswithcode.com/dataset/pascal-voc
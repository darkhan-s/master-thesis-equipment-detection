# Equipment identification through image recognition
## Master's thesis project repository of Saidnassimov Darkhan 
### Automation and Electrical Engineering, Aalto University
### Issued by Metso:Outotec

#### The following is the abstract of the thesis:
Object detection is a rapidly-evolving field with applications varying from medicine to self-driving vehicles. As the performance of the deep learning algorithms grow exponentially, countless object detection applications have emerged. Despite the nearly all-time high demand, object detection is rarely used in industrial applications. Historically, object detection requires extensive training data in order to produce sufficient results. Collecting huge datasets is often impractical in an  industrial environment due to the confidentiality restrictions and data accessibility limitations. 

This thesis attempts to minimize the manual labeling process by proposing a regularized cross-domain adaptive teacher model with continual learning. The model assumes a task that seeks to eliminate the domain shift between industrial datasets: a larger labeled dataset of rendered images and a smaller unlabeled dataset of real-life images. While the labels for the rendered images can be generated automatically, only a tiny amount of real images needs to be collected, which is crucial for the system scalability in industrial environments. The model transfers knowledge from one domain to another by means of adversarial domain adaptation and mean teacher training. In an attempt to achieve state-of-the-art results, this thesis proposes to regularize the student and the teacher networks using image-and instance-level alignment as well as consistency loss. Additionally, the model adopts a lifelong learning approach with network expansion and gradient regularization that enables the model to be retrainable on a continuously expanding dataset, which further facilitates the scalablity of the system.

As a result, the proposed  Adaptive teacher model with two-level alignment achieved competitive results with AP50 = 69.57 % at 14 999 iterations, which is twice as fast compared to the original model with AP50 = 71.40 % at 30 999 iterations. On the other hand, the continual learning experiment with 10 arbitrary classes proved that retraining the model on the entire dataset (AP50 = 63.72 %) brings more benefit than training the model continuously using the proposed approach (AP50 = 45.56 %). Finally, the proposed model was evaluated on one Metso Outotec equipment item, which included 1000 labeled rendered images and 28 unlabeled real images. The tests achieved a fair performance of AP50 = 85.86 %.  


#### Folder breakdown
The directory [docs](docs/) contains all the documentation as well as the LaTex code to produce the final thesis pdf. The images in this directory were used in the thesis and were cited in the produced pdf file. The directory [dataset_format_template](dataset_format_template/) contains the recommended dataset format. The original dataset used in the experiments can be found [here](http://cmp.felk.cvut.cz/t-less/download.html). However, it was converted using the [tools](tools/) and will be uploaded later. The example of the annotated rendered image is shown below:

![Labeled target image!](/demo.jpg "Labeled target image")

The directory [src](src/) contains the implementation of the regularized cross-domain model with continual learning. The model was mostly based on the implementations of the following papers: [Cross-domain adaptive teacher](https://github.com/facebookresearch/adaptive_teacher/) and [Exploring Categorical Regularization for Domain Adaptive Object Detection](https://github.com/megvii-research/CR-DA-DET/).


A minimalistic UI was developed to showcase the performance of the model. The app is hosted on `localhost:5000` using Flask API. Prior to starting the demo, the model has to be trained as instructed in the [script](src/train_adapt_teacher_tless.sh). This script was meant for [csc.fi](csc.fi/) machines. The images can be uploaded to the demo app in .jpg or .png format. 

An example command for starting the demo: 
`python app.py --config-file "faster_rcnn_R101_cross_pump.yaml" --weights-file "/full_path_to_model/model_best.pth" --dataset_path "/full_path_to_dataset/" `

If everything is done correctly, the demo should produce the following result: 
![Predictions!](/static/predictions/IMG_8890.jpg "Predictions")

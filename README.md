# master-thesis-equipment-detection
Darkhan's Master's thesis repository. The project attempts to tackle the challenge of equipment recognition in an industrial environment using 3D models.

A minimalistic UI was developed to showcase the performance of the model. For now, the model that performs the best, according to the AP metrics, is loaded to this WebApp, which is hosted on localhost:5000 and uses Flask API. The images can be uploaded in .jpg or .png format. Example images will be added later. 

example command for starting demo: 
``` python app.py --config-file "faster_rcnn_R101_cross_pump.yaml" --weights-file "/scratch/project_2005695/mt-ed-deploy/output-mymodel-pumps-FINAL-MyModel_whCustomAugmentation/model_best.pth" --dataset_path "/scratch/project_2005695/master-thesis-equipment-detection/bin/pumps/" ```

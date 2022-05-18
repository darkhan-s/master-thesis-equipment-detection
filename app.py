import os
from flask import Flask, render_template,request, send_file,  flash, redirect, url_for

from werkzeug.utils import secure_filename
from Detector import Detector
import io
import sys
import numpy as np
from PIL import Image
import traceback
import cv2

UPLOAD_FOLDER = os.path.join('static', 'uploads')
PREDICTIONS_FOLDER = os.path.join('static', 'predictions')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
detector = Detector()

import logging
logging.basicConfig(level=logging.INFO)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


## Needs some clean up to remove boilerplate file format conversions etc
@app.route('/',methods=['GET','POST'])
def entry_point():
	print('Starting default process and awaiting request', file=sys.stdout)

	if request.method=='GET':
		return render_template('index.html')
	if request.method=='POST':
		print('POST request received', file=sys.stdout)
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		try:
			file=request.files['file']
			# if user does not select file, browser also
			# submit a empty part without filename
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)

			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#image=file.read()
			
			#return render_template('result.html',result=prediction)
			image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			# buffer = io.BytesIO()
			# file.save(buffer)

			# image = np.array(Image.open(buffer, formats=["JPEG", "PNG"]))

			print('File read, attempting to run inference', file=sys.stdout)
			# run inference
			result_img = run_inference(image)
			clone_image = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(app.config['PREDICTIONS_FOLDER'], file.filename), clone_image)
			# create file-object in memory
			file_object = io.BytesIO()

			# write PNG in file-object
			result_img.save(file_object, 'PNG')

			# move to beginning of file so `send_file()` it will read from start    
			file_object.seek(0)

			print('Returning the predictions..', file=sys.stdout)
			#return send_file(file_object, mimetype='image/PNG')
			
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
			full_predictions_filename = os.path.join(app.config['PREDICTIONS_FOLDER'], file.filename)
			return render_template("index.html", user_image = full_filename, return_image = full_predictions_filename)	
			
		except Exception:
			print(traceback.format_exc(), file=sys.stdout)
			return render_template('error.html')


# run inference using detectron2
def run_inference(image):

	# run inference using detectron2
	result_img = detector.predict(image)

	# clean up
	try:
		os.remove(image)
	except:
		pass

	return result_img


if __name__ == '__main__':
	app.run(debug=True,port=os.getenv('PORT',5000))

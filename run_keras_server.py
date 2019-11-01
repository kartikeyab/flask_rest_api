from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
# initialize our Flask application and the Keras model

MODEL_PATH = 'model_weights/model10_latestversion.h5'
app = flask.Flask(__name__)
model = None

def load_saved_model():
	global model
	model = load_model(MODEL_PATH)
	global graph
	graph = tf.get_default_graph()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = image.astype('float32')
	image /= 255
	image = np.expand_dims(image, axis=0)

	image = image.reshape(-1,50, 50, 3)
	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	result = {}
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(50, 50))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			graph = tf.get_default_graph()
			with graph.as_default():
				preds = model.predict(image)
				preds_sub = preds.tolist()
				if preds_sub[0][0]<0.5:
					result['Predicted Class']=str('Clip')
				else:
					result['Predicted Class']=str('Noclip')

	# return the data dictionary as a JSON response
	return flask.jsonify(result)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_saved_model()
	app.run()

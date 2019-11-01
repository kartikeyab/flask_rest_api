from run_keras_server import load_saved_model,app

load_saved_model()
app.run()

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
			#graph = tf.get_default_graph()
			with graph.as_default():
				preds = model.predict(image)
				preds_sub = preds.tolist()
				if preds_sub[0][0]<0.5:
					result['Predicted Class']=str('Clip')
				else:
					result['Predicted Class']=str('Noclip')

	# return the data dictionary as a JSON response
	return flask.jsonify(result)

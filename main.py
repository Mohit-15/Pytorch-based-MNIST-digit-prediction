
from flask import Flask, jsonify, request
from utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTN = ('png', 'jpg', 'jpeg')
def allowed_file(filename):
	## example: abc.png
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTN


@app.route('/predict', methods=['POST'])
def predict():
	# 1. Loading the Image
	# 2. Image -> Tensor
	# 3. prediction
	# 4. return json

	if request.method == 'POST':
		file = request.files.get('file')

		if file is None or file.filename == '':
			return jsonify({'error': "No file."})

		if not allowed_file(file.filename):
			return jsonify({'error': 'Format not supported.'})

		try:
			image_byte = file.read()
			tensor = transform_image(image_byte)
			prediction = get_prediction(tensor)
			data = {
				'prediction': prediction.item(), 
				'class_name': str(prediction.item())
			}
			return jsonify(data)

		except:
			return jsonify({'error': 'error during prediction.'})

	return jsonify({'message': 'Prediction API is working. Use POST method to get predictions.'})
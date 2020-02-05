from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
app = Flask(__name__)

#ライセンス
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp', 'PNG', 'JPG', 'JPEG', 'GIF', 'BMP'])
IMAGE_WIDTH = 600
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(file_name, input_name)
	if file_name.endswith(".png"):
		image_reader = tf.image.decode_png(
			file_reader, channels=3, name="png_reader")
	elif file_name.endswith(".gif"):
		image_reader = tf.squeeze(
			tf.image.decode_gif(file_reader, name="gif_reader"))
	elif file_name.endswith(".bmp"):
		image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
	else:
		image_reader = tf.image.decode_jpeg(
			file_reader, channels=3, name="jpeg_reader")
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0)
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)
	return result

def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
	if request.method == 'POST':
		img_file = request.files['img_file']
		if img_file and allowed_file(img_file.filename):
			filename = secure_filename(img_file.filename)
		else:
			return render_template('error.html')

		f = img_file.stream.read()
		bin_data = io.BytesIO(f)
		file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		raw_img = cv2.resize(img, (IMAGE_WIDTH, int(IMAGE_WIDTH*img.shape[0]/img.shape[1])))

		raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
		cv2.imwrite(raw_img_url, raw_img)

		if __name__ == "__main__":
			images = raw_img_url
			file_name = images	#推論する画像
			model_file = "output_graph.pb"	#学習データ
			label_file = "penguin_list.txt"	#学習データのラベル
			input_height = 299
			input_width = 299
			input_mean = 0
			input_std = 255
			input_layer = "Placeholder"
			output_layer = "final_result"

			graph = load_graph(model_file)
			t = read_tensor_from_image_file(
				file_name,
				input_height=input_height,
				input_width=input_width,
				input_mean=input_mean,
				input_std=input_std)

			input_name = "import/" + input_layer
			output_name = "import/" + output_layer
			input_operation = graph.get_operation_by_name(input_name)
			output_operation = graph.get_operation_by_name(output_name)

			with tf.Session(graph=graph) as sess:
				results = sess.run(output_operation.outputs[0], {
					input_operation.outputs[0]: t
				})
			results = np.squeeze(results) * 100	#100%表示にする
			
			top_k = results.argsort()[-1:][::-1]
			debug = top_k
			debug = results.argsort()[-5:][::-1]
			labels = load_labels(label_file)
			result = ""
			for i in top_k:
				result = labels[i] + "ペンギンの確立：" + str(results[i]) + "%"
			print(result)
			for j in debug:
				print(labels[j] + "ペンギン → " + str(results[j]) + "%")

		return render_template('result.html', result = result, raw_img_url=raw_img_url)

	else:
		return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
	app.debug = True
	app.run(host="0.0.0.0",port=80)
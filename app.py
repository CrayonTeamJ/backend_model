from flask import Flask, jsonify, make_response, request
import requests
from flask_pymongo import PyMongo
from function.to_frame import *
from detect import *

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://team-j-test01:team-j-test01@cluster0.ybzxz.mongodb.net/sample_restaurants?retryWrites=true&w=majority"
mongo = PyMongo(app)


@app.route('/to_yolo', methods = ['POST'])
def user_only():
    video_info = request.json

    #data = {'video_pk': pk, 's3_video': video_path}
    video_pk = request.form['video_pk']
    video_path = reuqest.form['s3_video']

    video_to_Img(video_path)
    result = run(source='backend_model/data/images')
    

    return make_response(jsonify({'pk': video_info['video_pk'], 's3_path': video_info['s3_video']}), 200)

    
#flask run --host=0.0.0.0 --port=5001


from flask import Flask, jsonify, make_response, request
import requests
from flask_pymongo import PyMongo
from function.to_frame import *
from detect import *
import ffmpeg
import shutil

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://team-j-test01:team-j-test01@cluster0.ybzxz.mongodb.net/sample_restaurants?retryWrites=true&w=majority"
mongo = PyMongo(app)


@app.route('/to_yolo', methods = ['POST'])
def user_only():
    video_info = request.json

    #data = {'video_pk': pk, 's3_video': video_path}
    video_pk = request.form['video_pk']
    video_path = request.form['s3_video']

    video_to_Img(video_path)
    result = run(source='backend_model/data/images')
    return make_response(jsonify({'pk': video_info['video_pk'], 's3_path': video_info['s3_video']}), 200)

    
#flask run --host=0.0.0.0 --port=5001


app.config["MONGO_URI"] = "mongodb+srv://Crayon:pc2Af0vKZWbkT7GL@clustercrayon.lij0j.mongodb.net/voicedb?retryWrites=true&w=majority"
mongodb_client = PyMongo(app)
coll = mongodb_client.db.video_files_list


@app.route('/to_yolooo', methods = ['POST'])
def run_yolo():
    req_data = request.json
    video_pk = req_data['video_pk']
    video_path = req_data['video_path']
    dir = 'backend_model/data/{number}/image-%3d.jpg'.format(number=video_pk)
    # video_to_Img(video_path, video_pk)
    ffmpeg.input(video_path).filter('fps', fps='1').output(dir, start_number=0, **{'qscale:v': 3}).overwrite_output().run(quiet=True)
    dir2 = 'backend_model/data/{number}'.format(number=video_pk)
    return dir2
    # result = run(source=dir2)
    # coll.insert({'video_number':video_pk, 'detection_list':result})
    # 아마존 s3에 업로드하는 코드 한줄(dir2)
    # ## shutil.rmtree(dir2)
    # return make_response(jsonify({'Result': 'Success', 'video_pk': video_pk}), 200)


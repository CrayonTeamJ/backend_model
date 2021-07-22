from flask import Flask, jsonify, make_response, request
import requests
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_pymongo import PyMongo
from function.to_frame import *
from detect import *
from function.s3_control import *
import ffmpeg
import shutil
import config

app = Flask(__name__)
db = SQLAlchemy()
migrate = Migrate()

app.config.from_object(config)
db.init_app(app)
migrate.init_app(app, db)
app.config["MONGO_URI"] = "mongodb+srv://team-j-test01:team-j-test01@cluster0.ybzxz.mongodb.net/sample_restaurants?retryWrites=true&w=majority"
mongo = PyMongo(app)


# @app.route('/to_yolo', methods = ['POST'])
# def user_only():
#     video_info = request.json

#     #data = {'video_pk': pk, 's3_video': video_path}
#     video_pk = request.form['video_pk']
#     video_path = request.form['s3_video']

#     video_to_Img(video_path)
#     result = run(source='backend_model/data/images')
#     return make_response(jsonify({'pk': video_info['video_pk'], 's3_path': video_info['s3_video']}), 200)

    
#flask run --host=0.0.0.0 --port=5001


app.config["MONGO_URI"] = "mongodb+srv://Crayon:pc2Af0vKZWbkT7GL@clustercrayon.lij0j.mongodb.net/voicedb?retryWrites=true&w=majority"
mongodb_client = PyMongo(app)
coll = mongodb_client.db.video_files_list


@app.route('/to_yolo', methods = ['POST'])
def run_yolo():
    req_data = request.json
    video_pk = req_data['video_pk']
    video_path = req_data['video_path']

    dir = 'data/'+str(video_pk)
    os.mkdir(dir)
    output_name = dir+'/image-%3d.jpg'
    video_to_Img(video_path, video_pk,output_name)
    # ffmpeg.input(video_path).filter('fps', fps='1').output(output_name, start_number=0, **{'qscale:v': 3}).overwrite_output().run(quiet=True)
    result = run(source=dir)
    coll.insert({'video_number':video_pk, 'detection_list':result})

    list_dir = []
    list_dir = os.listdir(dir)
    
    for filename in list_dir:
        # upload local image files to gcp storage
        upload_blob_file(dir + '/'+ filename, 'images/'+ str(video_pk)+ '/'+ filename)
        
        # delete local image files
    
    shutil.rmtree(dir)


    return result

    # 아마존 s3에 업로드하는 코드 한줄(dir2)
    # return make_response(jsonify({'Result': 'Success', 'video_pk': video_pk}), 200)


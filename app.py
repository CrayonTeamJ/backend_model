from flask import Flask, jsonify, make_response, request
import requests
from flask_pymongo import PyMongo
from function.to_frame import *
from collections import OrderedDict
from detect import *
from function.s3_control import *
import ffmpeg
import shutil

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://Crayon:pc2Af0vKZWbkT7GL@clustercrayon.lij0j.mongodb.net/voicedb?retryWrites=true&w=majority"
mongodb_client = PyMongo(app)
coll = mongodb_client.db.video_files_list
coll2 = mongodb_client.db.images_coll


@app.route('/to_yolo', methods = ['POST'])
def run_yolo():
    req_data = request.json
    video_pk = req_data['video_pk']
    video_path = req_data['video_path']

    dir = 'data/'+str(video_pk)
    os.makedirs(dir)
    output_name = dir+'/image-%3d.jpg'
    video_to_Img(video_path, video_pk, output_name)
    
    #run Yolo
    result = run(source=dir)

    #save result to mongoDB
    coll.insert({'video_number':video_pk, 'detection_list':result})

    #save images to S3
    list_dir = []
    list_dir = os.listdir(dir)

    count = 0
    output2=OrderedDict()
    image_list=[]

    for filename in list_dir:
        # upload local image files to gcp storage
        upload_blob_file(dir + '/'+ filename, 'images/'+ str(video_pk)+ '/'+ filename)

        #url form
        img_path = 'https://teamj-data.s3.ap-northeast-2.amazonaws.com/images/'+str(video_pk)+'/'+filename
        image_list.append({'time':count, 'path':img_path})
        count+=1
    
    output2["video_pk"]=video_pk
    output2["image_list"]=image_list
    coll2.insert(output2)

    # delete local directory
    shutil.rmtree(dir)

    return make_response(jsonify({'Result': 'Success', 'video_pk': video_pk}), 200)



#flask run --host=0.0.0.0 --port=5001

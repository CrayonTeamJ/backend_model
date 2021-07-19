from flask import Flask, jsonify, make_response, request
import requests

app = Flask(__name__)

@app.route('/to_yolo', methods = ['POST'])
def user_only():
    video_info = request.json
    return make_response(jsonify({'pk': video_info['video_pk'], 's3_path': video_info['s3_video']}), 200)

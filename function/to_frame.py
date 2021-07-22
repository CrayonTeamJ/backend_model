import subprocess
import os
import ffmpeg

def video_to_Img(video_path, video_pk, output_name):
    
    try:
        ffmpeg.input(video_path).filter('fps', fps='1').output(output_name, start_number=0, **{'qscale:v': 2}).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))


# video_to_Img('https://teamj-data.s3.ap-northeast-2.amazonaws.com/video/bts_colbert.mp4')
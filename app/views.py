import re
from app import app
from flask import render_template, request, redirect
import logging
import shutil
import os
import json
import subprocess
import numpy as np
from data_gen.preprocess import pre_normalization

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """

# @app.route("/profile/<username>")
# def profile(username):
#     return render_template("profile.html")

# app.config["IMAGE_UPLOADS"] = os.getcwd() + "/app/static/img/uploads"

# @app.route("/", methods=["GET", "POST"])
@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:
            image = request.files["image"]  #name set as image in upload_image.html

            if image.filename == "":
                logging.warning("Image must have filename")
                return redirect(request.url)
            upload_type = image.filename.split('.')[-1]
            if upload_type in ['jpg', 'png']:
                print('-------------------image----------------------------')
                folder = os.getcwd() + "/app/static/img"
            if upload_type in ['mp4','avi']:
                print('-------------------video----------------------------')
                folder = os.getcwd() + "/app/static/vid"

            if os.path.isdir(folder):
                shutil.rmtree(folder)
                os.makedirs(folder)
            else:
                os.makedirs(folder)  # create new  dir

            image.save(os.path.join(folder,  image.filename))
            print("image saved")

            extract_skeletons(upload_type, image.filename)

            return redirect(request.url)  # redirect to another endpoint

    return render_template("upload_image.html" , skel_filename = 'test')
@app.route("/run_inference")
def run_inference():
    print('IN RUN INFERENCE')
    np_generated = gen_numpy()
    if np_generated:
        command = './data_gen/youtube_gen_bone.py'
        subprocess.call(['python.exe', command], shell=True)
        print('Generated bone data!')

        command = './data_gen/youtube_gen_label.py'
        subprocess.call(['python.exe', command], shell=True)
        print('Generated labels!')

        command = './dgnn/main.py'
        print(os.getcwd())
        subprocess.call(['python.exe', command ,"--config", "./dgnn/config/nturgbd-cross-view/test_spatial.yaml"], shell=True)
    return render_template('upload_image.html')

def extract_skeletons(upload_type,filename ):  # must normalize x,y to [-1,1] first (referencing DGNN repo)
    if upload_type in ['jpg', 'png']:
        command = "./build/bin/OpenPoseDemo.exe --image_dir ./app/static/img --write_json ./app/static/skeleton --write_images ./app/static/resultimg"
    if upload_type in ['mp4','avi']:
        command = f"./build/bin/OpenPoseDemo.exe --video ./app/static/vid/{filename} --write_json ./app/static/skeleton  --write_video ./app/static/resultvid/{filename.split('.')[0]}.avi"  #

    ret = subprocess.run(command, capture_output=True)
    print(ret.stdout.decode())

def read_openpose_data(file):
    with open(file,'r') as f:
        frame_data = json.load(f)
    return frame_data
# print(read_openpose_data('/content/drive/MyDrive/openpose-keypoints/zjZbQ5QqCHU/video_000000000001_keypoints.json'))

def get_nonzero_std(s):
    # `s` has shape (T, V, C)
    # Select valid frames where sum of all nodes is nonzero
    s = s[s.sum((1,2)) != 0]
    if len(s) != 0:
        # Compute sum of standard deviation for all 3 channels as `energy`
        s = s[..., 0].std() + s[..., 1].std()
    else:
        s = 0
    return s

def gen_numpy():
    skeleton = []
    data_path = os.path.join(os.getcwd(), "app/static/skeleton")
    for frame in os.listdir(data_path):
        skeleton.append(read_openpose_data(os.path.join(data_path, frame)))

    max_body = 4
    max_body_true=2
    num_joint = 25
    max_frame = 300
    data = np.zeros((max_body, len(skeleton), num_joint, 3)) # (M,T,V,C)
    # print(f"data shape {data.shape}")

    for n, f in enumerate(skeleton):
        for m, b in enumerate(f.get('people')):
            joints = b.get('pose_keypoints_2d')
            for j in range(len(joints)):
                if m < max_body and j < num_joint:
                    if joints:
                        # print(f'{n},{m},{j}')
                        v = joints[:3]
                        data[m, n, j, :] = [v[0], v[1], 0]
                        del joints[:3]
                    else:
                        continue

    # print(f"data shape {data.shape}")  # data shape (4, 17, 25, 2)

    energy = np.array([get_nonzero_std(x) for x in data])
    # print(energy)   #[761.88276953 741.77521164 532.44723716   0.        ]
    index = energy.argsort()[::-1][0:max_body_true]
    # print(index)
    data = data[index]
    # print(data.shape)
    # print(data)

    data = data.transpose(3, 1, 2, 0) # (M, T, V, C) to (C, T, V, M)

    # Standardizing data
    # data = (data - data.mean())/(data.std())

    fp = np.zeros((1, 3, max_frame, 25, max_body_true), dtype=np.float32)

    fp[0, :, :data.shape[1], :, :] = data
    print(fp.shape)
    fp = pre_normalization(fp)
    print(fp)

    # print(fp.shape)  # (1, 2, 300, 25, 2)

    OUT_PATH = os.path.join(os.getcwd(), "dgnn/data/")
    if os.path.isdir(OUT_PATH):
        shutil.rmtree(OUT_PATH)
        os.makedirs(OUT_PATH)
    else:
        os.makedirs(OUT_PATH)  # create new  dir
    np.save(f'{OUT_PATH}/test_data_joint.npy' , fp)
    return True
    # pred_np = np.load(f'{OUT_PATH}/test_data_joint.npy')  # to test if np save successful
    # print(f'prednp {pred_np}'   )
    # print(pred_np.shape)


    #TODO: gen bone + label
    #TODO: try isolate and run inference model (CPU)

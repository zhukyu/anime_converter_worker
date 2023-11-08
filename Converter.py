# @title Define functions
# @markdown Select model version and run.
from tqdm import tqdm
from glob import glob
import os
import requests
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets
import onnxruntime as ort
import time
import cv2
import PIL
import numpy as np
from tqdm.notebook import tqdm
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import requests
import os
import time  # Simulating processing time
from flask import Flask, request
import threading
import websockets
import asyncio
import json
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
from boto3.s3.transfer import S3Transfer

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

django_ws = None

# Retrieve variables from .env file
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
CLOUDFRONT_DOMAIN = os.getenv('CLOUDFRONT_DOMAIN')
WEB_SOCKET_URI = os.getenv('WEB_SOCKET_URI')

vid_form = ['.mp4', '.avi', '.webm']
device_name = ort.get_device()

print("ONNX Runtime Version:", ort.__version__)
if device_name == 'cpu':
    print('CPU found')
    # providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    print('GPU found')
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        # 'CPUExecutionProvider',
    ]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
sess_options = ort.SessionOptions()

# @param ['AnimeGAN_Hayao','AnimeGANv2_Hayao','AnimeGANv2_Shinkai','AnimeGANv2_Paprika']
model = 'AnimeGANv2_Hayao'
# load model
# session = ort.InferenceSession(f'{model}.onnx', sess_options=sess_options, providers=providers)
ort_providers = ["CUDAExecutionProvider"]
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    f'{model}.onnx', options, providers=ort_providers)

print("Session Providers: ", session.get_providers())


def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_S3_REGION_NAME)
    transfer = S3Transfer(s3)
    try:
        print("Uploading to S3: ", s3_file)
        transfer.upload_file(local_file, bucket, s3_file)
        # s3.upload_file(local_file, bucket, s3_file)
        s3_path = f"https://{CLOUDFRONT_DOMAIN}/{s3_file}"
        print("Upload Successful: ", s3_path)
        return s3_path
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None


def post_precess(img, wh):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8).clip(0, 255)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img


def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32:  # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img


def Convert(img, scale):
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name
    fake_img = session.run(None, {x: img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, scale[::-1])
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)


in_dir = '/content/in'
out_dir = f"/content/outputs"


def get_video(video, out_name, output_format='MP4V'):

    download_url = None

    # load video
    vid = cv2.VideoCapture(video)
    vid_name = os.path.basename(video)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*output_format)

    video_out = cv2.VideoWriter(out_name, codec, fps, (width, height))
    pbar = tqdm(total=total, )
    pbar.set_description(
        f"Making: {os.path.basename(video).rsplit('.', 1)[0] + '_converted.mp4'}")

    frame_count = 0  # A counter for the current frame

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = np.asarray(np.expand_dims(process_image(frame), 0))
        x = session.get_inputs()[0].name
        y = session.get_outputs()[0].name
        fake_img = session.run(None, {x: frame})[0]
        fake_img = post_precess(fake_img, (width, height))
        video_out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        pbar.update(1)
        frame_count += 1

        # Calculate processing progress
        # progress = (frame_count / total) * 100

        # Yield progress
        yield frame_count, total, download_url

    pbar.close()
    vid.release()
    video_out.release()
    print(f"Video saved to: {out_name}")

    # upload to s3
    s3_file = f"outputs/{os.path.basename(out_name)}"
    s3_path = upload_to_s3(out_name, AWS_STORAGE_BUCKET_NAME, s3_file)
    download_url = s3_path

    yield frame_count, total, download_url


def download_video(video_url, save_path):
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def processURL(video_url, video_id, video_name, output_directory='./outputs'):

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Download the video to a temporary file
    temp_video_path = 'in/' + video_name
    print(f"Downloading video to: {temp_video_path}")
    download_video(video_url, temp_video_path)

    # Validate the video file format
    vid_form = ['.mp4', '.avi', '.webm']
    if os.path.splitext(temp_video_path)[1].lower() not in vid_form:
        print(f"Unsupported video format: {temp_video_path}")
        return

    # Determine the output video file name
    out_name = f"{output_directory}/{os.path.basename(temp_video_path).split('.')[0]}_converted.mp4"

    # Process the video and track the progress
    video_processor = get_video(temp_video_path, out_name)

    # Set up the event loop for this thread if necessary
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        for frame_count, total, download_url in video_processor:
            # Run the async function on the event loop
            loop.run_until_complete(send_progress_to_django(
                frame_count, total, video_id, download_url))
    finally:
        loop.close()

    # Optionally, you can remove the temporary video file after processing
    os.remove(temp_video_path)

# processURL('https://d2v4nc4g5kecxn.cloudfront.net/videos/AT-cm_CEmWzGoG1wSvv0Yi_vNivg+(1).mp4')


async def handle_message(video_url, video_id, video_name):
    print(f"Received video URL: {video_url}")
    # Process the video and send progress updates
    threading.Thread(target=processURL, args=(
        video_url, video_id, video_name), daemon=True).start()


async def send_progress_to_django(frame_count, total, video_id, download_url):
    progress_data = json.dumps({
        'type': 'progress',
        'progress': frame_count,
        'total_size': total,
        'video_id': video_id,
        'download_url': download_url
    })
    print(f"Sending progress: {progress_data}")
    if django_ws:
        await django_ws.send(progress_data)


def start_django_ws_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(django_ws_client())
    loop.run_forever()


async def django_ws_client():
    global django_ws
    uri = WEB_SOCKET_URI  # Django WebSocket URL
    async with websockets.connect(uri) as websocket:
        django_ws = websocket
        try:
            while True:  # Keep the connection open to receive and send messages
                message = await websocket.recv()
                data = json.loads(message)
                video = data['message']
                await handle_message(video['video_url'], video['video_id'], video['video_name'])
        finally:
            django_ws = None

threading.Thread(target=start_django_ws_client, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

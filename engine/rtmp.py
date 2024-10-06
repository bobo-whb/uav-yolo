import asyncio
import json
import queue
import select
import subprocess
import threading
import time

import cv2

from cfg.config import MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, FPS, GOP_SIZE
from engine.mqtt import MQTTClientHandler


def init_rtmp_command(width, height, rtmp_url, fps=FPS, gop_size=GOP_SIZE):
    ffmpeg_command = ['ffmpeg',
                      '-re',
                      '-y',
                      '-f', 'rawvideo',
                      '-vcodec', 'rawvideo',
                      '-pix_fmt', 'bgr24',
                      '-s', "{}x{}".format(width, height),
                      '-r', str(fps),
                      '-i', '-',
                      '-c:v', 'libx264',
                      '-pix_fmt', 'yuv420p',
                      '-preset', 'ultrafast',
                      '-g', str(gop_size),
                      '-f', 'flv',
                      '-flvflags', 'no_duration_filesize',
                      '-rw_timeout', '1000000',
                      rtmp_url]
    return ffmpeg_command


def init_rtmp_stream(ffmpeg_command):
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, bufsize=0)
    _, writable, _ = select.select([], [process.stdin], [], 0)
    return process, writable


class VideoCapture:
    def __init__(self, url, rtmp_url, custom_frame_processor):
        self.isstop = False
        self.q = queue.Queue()
        self.init_event = threading.Event()
        self.size = None
        self.receive_thread = threading.Thread(target=self.readframe, args=(url,))
        self.process_thread = threading.Thread(target=self.process_frames, args=(custom_frame_processor, rtmp_url))
        self.receive_thread.start()
        self.process_thread.start()
        print('VideoCapture started!')

    def release(self, custom_release=None):
        self.isstop = True
        if custom_release is not None:
            custom_release()
        print('VideoCapture stopped!')

    def readframe(self, url):
        capture = cv2.VideoCapture(url)
        while True:
            if capture.isOpened() or self.isstop:
                print("Video source opened!")
                break
            print(f"Unable to open video source {url}. Retrying...")
            capture.release()
            capture = cv2.VideoCapture(url)
        self.size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.init_event.set()
        while not self.isstop:
            start = time.time()
            ok, frame = capture.read()
            if ok:
                self.q.put((frame, int(capture.get(cv2.CAP_PROP_POS_FRAMES)), capture.get(cv2.CAP_PROP_FPS)))
            else:
                end = time.time()
                if end - start > 20:
                    while True:
                        print(f"Unable to open video source {url}. Retrying...")
                        capture.release()
                        capture = cv2.VideoCapture(url)
                        if capture.isOpened() or self.isstop:
                            print("Video source opened!")
                            break
        capture.release()
        print("reading stopped!")

    def process_frames(self, frame_callback, rtmp_url):
        self.init_event.wait()
        ffmpeg_command = init_rtmp_command(self.size[0], self.size[1], rtmp_url)
        ffmpeg_process, writable = init_rtmp_stream(ffmpeg_command)
        while not self.isstop or not self.q.empty():
            if self.q.empty():
                continue
            frame = self.q.get_nowait()
            frame = frame_callback(frame)
            while not self.isstop or not self.q.empty():
                if ffmpeg_process.poll() is not None:
                    ffmpeg_process.kill()
                    ffmpeg_process.communicate()
                    ffmpeg_process, writable = init_rtmp_stream(ffmpeg_command)
                    print("FFMPEG process restarted!")
                if ffmpeg_process.stdin in writable:
                    try:
                        ffmpeg_process.stdin.write(frame.tobytes())
                        break
                    except BrokenPipeError:
                        print("Broken pipe error occurred!")
        ffmpeg_process.kill()
        ffmpeg_process.communicate()
        print("processing stopped!")


if __name__ == '__main__':
    cap = None


    def frame_processor(frame):
        time.sleep(0.04)
        return frame[0]


    def on_message(client, topic, payload, qos, properties):
        print(f"Received message: {payload.decode()} on topic {topic}")
        data = json.loads(payload.decode())
        if "param" in data and "resource" in data:
            global cap
            cap = VideoCapture(data["resource"], data["result"], frame_processor)
        elif "flightId" in data and len(data) == 1:
            if cap is not None:
                cap.release()


    client_handler = MQTTClientHandler(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
                                       on_message_callback=on_message)
    asyncio.run(client_handler.run())

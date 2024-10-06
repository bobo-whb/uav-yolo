import json
import multiprocessing
import asyncio
import threading

from engine.mqtt import MQTTClientHandler
from cfg.config import TRACKER_CONFIG, MODEL_PATH, MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD
from tracker import VideoObjectTracker
from engine.rtmp import VideoCapture


stop_event_dict = {}


def run_videotracker(data, stop_event):
    def wait_for_stop():
        stop_event.wait()
        cap.release(tracker.stop_detection())

    tracker = VideoObjectTracker(
        model_path=MODEL_PATH,
        tracker_config=TRACKER_CONFIG,
        output_path="",
        uav_height=80,
        angle=40,
        zoom=2,
    )
    tracker.update_parameters(
        uav_height=data["param"]["height"],
        angle=data["param"]["pith"],
        zoom=data["param"]["zoom"],
        flight_id=data["flightId"],
        mqtt_client=client_handler,
    )

    cap = VideoCapture(data["resource"], data["result"], tracker.process_frames)
    wait_thread = threading.Thread(target=wait_for_stop)
    wait_thread.start()

    cap.process_thread.join()
    cap.receive_thread.join()
    wait_thread.join()
    send_complete_message(data["flightId"])


def on_message(client, topic, payload, qos, properties):
    print(f"Received message: {payload.decode()} on topic {topic}")
    data = json.loads(payload.decode())
    flight_id = data.get("flightId")

    if "param" in data and "resource" in data:
        if flight_id:
            stop_event = multiprocessing.Event()
            process = multiprocessing.Process(target=run_videotracker, args=(data, stop_event))
            process.start()
            stop_event_dict[flight_id] = stop_event
    elif "flightId" in data and len(data) == 1:
        if flight_id in stop_event_dict:
            stop_event = stop_event_dict[flight_id]
            stop_event.set()
            del stop_event_dict[flight_id]

def send_complete_message(flightId):
    topic = f"/ATS/yunying/task/ai/out/FLXJ"
    message = json.dumps({"flightId": flightId})
    try:
        client_handler.publish(topic, message)
    except Exception as e:
        print(f"Failed to publish complete message: {e}")

if __name__ == "__main__":
    client_handler = MQTTClientHandler(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
                                       on_message_callback=on_message)
    asyncio.run(client_handler.run())

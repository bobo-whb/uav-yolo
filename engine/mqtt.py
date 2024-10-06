import asyncio
import uuid
from gmqtt import Client as MQTTClient


def default_on_message(client, topic, payload, qos, properties):
    print('MSG:', payload)


def on_connect(client, flags, rc, properties):
    print("Connected to MQTT Server")
    client.subscribe("/ATS/yunying/task/ai/in/FLXJ", qos=0)


def on_publish(client, mid, qos, properties):
    print(f"Message {mid} has been published with QoS {qos}")


class MQTTClientHandler:
    def __init__(self, host, port, username, password, on_message_callback=None):
        self.client = MQTTClient(str(uuid.uuid1()))
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.on_message_callback = on_message_callback or default_on_message
        self._setup_callbacks()

    def _setup_callbacks(self):
        self.client.on_connect = on_connect
        self.client.on_message = self.on_message_callback  # 使用外部传入的回调
        self.client.on_publish = on_publish

    async def connect(self):
        self.client.set_auth_credentials(self.username, self.password)
        await self.client.connect(self.host, self.port)

    def publish(self, topic, payload, qos=0):
        self.client.publish(topic, payload, qos=qos)

    async def run(self):
        await self.connect()
        await asyncio.Event().wait()


if __name__ == '__main__':
    # 自定义的 on_message 回调函数
    def custom_on_message(client, topic, payload, qos, properties):
        print('Custom MSG:', payload)


    # 创建时传入自定义的 on_message 回调
    client_handler = MQTTClientHandler('12.30.4.63', 42417, 'admin', 'public', on_message_callback=custom_on_message)

    asyncio.run(client_handler.run())

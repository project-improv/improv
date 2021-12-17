import time
import zmq
import random
from improv.actor import Actor
from demos.live.actors.acquire_zmq import ZMQAcquirer

class ZMQSender(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

    def setup(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5000")
        i = 1
        while True:
            msg = "Hi for the %d:th time..." % i
            socket.send_string(msg)
            print("Sent string: %s ..." % msg)
            i += 1
            time.sleep(1)

if __name__ == "__main__":
    zmq_sender = ZMQSender("zmq_sender")






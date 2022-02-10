import time
import zmq
from improv.actor import Actor
import os
import numpy as np

class ZMQSender(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

    def setup(self):
        file_path = os.getcwd()
        file_path = file_path[0: len(file_path) - 6] + "/Pandas3D/stimmed.txt"
        arr = np.loadtxt(file_path)

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:1234")

        for stimulus in arr:
            self.send_array(socket, A=stimulus)
            time.sleep(2)

    def send_array(self, socket, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        print(A)
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        socket.send_json(md, flags | zmq.SNDMORE)
        print("sent!")
        return socket.send(A, flags, copy=copy, track=track)

if __name__ == "__main__":
    zmq_sender = ZMQSender("zmq_sender")






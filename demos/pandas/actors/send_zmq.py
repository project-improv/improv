import time
import zmq
from improv.actor import Actor, RunManager
import os
import numpy as np

class ZMQSender(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:1234")

    def run(self):
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []
        with RunManager(self.name, self.sendZMQ, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

    def sendZMQ(self):
        file_path = os.getcwd()
        print(file_path)
        file_path = file_path + "/Pandas3D/stimmed.txt"
        arr = np.loadtxt(file_path)
        for stimulus in arr:
            self.send_array(self.socket, A=stimulus)
            time.sleep(2)

    def send_array(self, socket, A, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        print(A)
        md = dict(
            dtype=str(A.dtype),
            shape=A.shape,
        )
        socket.send_json(md, flags | zmq.SNDMORE)
        return socket.send(A, flags, copy=copy, track=track)

if __name__ == "__main__":
    zmq_sender = ZMQSender("zmq_sender")






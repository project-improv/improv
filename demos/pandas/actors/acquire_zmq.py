import zmq
from improv.actor import Actor, RunManager, Spike
import os
import numpy as np

class ZMQAcquirer(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("init")

    def setup(self):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect("tcp://127.0.0.1:1234")
        self.sock.subscribe("") # Subscribe to all topics

        self.saveArray = []
        self.save_ind = 0
        self.fullStimmsg = []

        self.tailF = False
        self.stimF = False
        self.frameF = False

    def run(self):
        print("Starting receiver loop ...")
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []
        with RunManager(self.name, self.runAcquirer, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)

    def runAcquirer(self):
        try:
            msg = self.recv_array(socket=self.sock)

            frame_number = float(msg[0])
            internal_index = float(msg[1])
            angle = float(msg[2])
            vel = float(msg[3])
            freq = float(msg[4])
            contrast = float(msg[5])
            arr = [frame_number, internal_index, angle, vel, freq, contrast]
            obj_id = self.client.put(arr, 'acq_pandas' + str(frame_number))
            self.q_out.put([str(frame_number), obj_id])

        except Exception as e:
            print('error: {}'.format(e))

    def recv_array(self, socket, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])

if __name__ == "__main__":
    zmq_acquirer = ZMQAcquirer("zmq_acquirer")



from improv.actor import Actor, RunManager
import numpy as np
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Processor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self):
        self.done = False
        self.name = "Processor"
        self.frame = None
        self.avg_list = []
        self.frame_num = 1

    def run(self):
        with RunManager(self.name, self.get_avg, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)
    

    def get_avg(self):
        # print("Getting Average!")
        frame = None
        print(f"Memory address of q_in: {hex(id(self.q_in))}")
        print(f"Memory address of processor.client: {hex(id(self.client))}")
        # print("Trying to get frame")
        try:
            print("Acquiring frame")
            frame = self.q_in.get(timeout=0.05)
            print(f"Acquired frame: {frame}")
        except:
            print("Could not get frame")

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame = self.client.getID(frame[0][0])
            print(f"self.frame: {self.frame}")
            avg = np.mean(self.frame[0])
            print(f"Average: {avg}") 
            self.avg_list.append(avg)
            print(f"Overall Average: {np.mean(self.avg_list)}")
            print(f"Frame number: {self.frame_num}")
            self.frame_num += 1
        
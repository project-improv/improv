from improv.actor import Actor, RunManager
import numpy as np
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Processor(Actor):
    """ Sample processor used to calculate the average of an array of integers.

    Intended for use with sample_generator.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self):
        """ Initializes all class variables.
        
        self.name (string): name of the actor.
        self.frame (ObjectID): Limbo object id referencing data from the store.
        self.avg_list (list): list that contains averages of individual vectors.
        self.frame_num (int): index of current frame.
        """

        self.name = "Processor"
        self.frame = None
        self.avg_list = []
        self.frame_num = 1

    def run(self):
        with RunManager(self.name, self.get_avg, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)
    

    def get_avg(self):
        """ Gets from the input queue and calculates the average.
        
        Receives an ObjectID, references data in the store using that
        ObjectID, calculates the average of that data, and finally prints 
        to stdout. 
        """

        frame = None
        try:
            frame = self.q_in.get(timeout=0.05)
        except:
            logger.error("Could not get frame!")
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame = self.client.getID(frame[0][0])
            avg = np.mean(self.frame[0])
            print(f"Average: {avg}") 
            self.avg_list.append(avg)
            print(f"Overall Average: {np.mean(self.avg_list)}")
            print(f"Frame number: {self.frame_num}")
            self.frame_num += 1
        
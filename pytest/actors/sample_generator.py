from improv.actor import Actor, RunManager
import numpy as np
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Generator(Actor):
    """ Sample actor to generate data to pass into a sample processor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.done = False
        self.name = "Generator"
        self.frame_num = 0

    def setup(self):
        """ Generate array.
        """

        print("setting up started")
        self.data = np.asmatrix(np.random.randint(100, size = (100, 5)))
        print(self.q_comm.empty())
        print("setting up done")

    def run(self):
        """ Send array into the store.
        """
        with RunManager(self.name, self.generate, self.setup, self.q_sig, self.q_comm) as rm:
            logger.info(rm)
        # send to store interface
        
    def generate(self):
        print("Running generate")
        if self.done:
            pass
        print(f"Mem addr of generator.q_out: {hex(id(self.q_out))}")
        print(f"Mem addr of generator.client: {hex(id(self.client))}")
        print(f"Frame num: {self.frame_num}")
        print(f"dims: {np.shape(self.data)}")
        if(self.frame_num < np.shape(self.data)[0]):
            print("Entered condition")
            print(f"Client Object: {self.client} \n Client Mem &: {hex(id(self.client))}")
            data_id = self.client.put(self.data[self.frame_num], str(f"Gen_raw: {self.frame_num}"))
            print(f"Data ID: {data_id}")
            try:
                print(f"Putting to q_out on frame number {self.frame_num}") 
                self.q_out.put([[data_id, str(self.frame_num)]])
                self.frame_num += 1
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
        else:
            self.data = np.concatenate((self.data, np.asmatrix(np.random.randint(10, size=(1, 5)))), axis=0)

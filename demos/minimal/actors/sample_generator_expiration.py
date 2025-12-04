from actors.sample_generator_zmq import Generator
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator_exp(Generator):
    """Sample actor to generate data to pass into a sample processor with TTL for each Redis key

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.name = "Generator_exp"
        self.frame_num = 0

    def runStep(self):
        """Generates additional data after initial setup data is exhausted.

        Data is of a different form as the setup data in that although it is
        the same size (5x1 vector), it is uniformly distributed in [1, 10]
        instead of in [1, 100]. Therefore, the average over time should
        converge to 5.5.
        """

        if self.frame_num < np.shape(self.data)[0]:
            if self.store_loc:
                data_id = self.client.put(
                    self.data[self.frame_num], str(f"Gen_raw: {self.frame_num}")
                )
            else:
                data_id = self.client.put(self.data[self.frame_num], ex=40)
            # logger.info('Put data in store')
            try:
                if self.store_loc:
                    self.q_out.put([[data_id, str(self.frame_num)]])
                else:
                    self.q_out.put(data_id)  # AsyncQueue.put()
                # logger.info("Sent message on")

                self.frame_num += 1
            except Exception as e:
                logger.error(
                    f"--------------------------------Generator Exception: {e}"
                )
        else:
            self.data = np.concatenate(
                (self.data, np.asmatrix(np.random.randint(10, size=(1, 5)))), axis=0
            )

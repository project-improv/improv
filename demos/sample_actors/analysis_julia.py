import logging
import time
from queue import Empty
import julia
import numpy as np
from nexus.actor import Actor, RunManager
from nexus.store import ObjectNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class JuliaAnalysis(Actor):
    """
    Class to run analyses in Julia.
    This actor puts in the data store the average frame intensity.
    """

    def __init__(self, *args, julia_file="julia_func.jl"):
        """julia_file: path to .jl file(s) for analyses computed in Julia"""

        super().__init__(*args)

        self.julia = None
        self.julia_file = julia_file
        self.j_func = list()

        self.frame = None  # np.ndarray
        self.frame_number = 0
        self.result_ex = None  # np.ndarray

        self.t_per_frame = list()
        self.t_per_put = list()

    def setup(self):
        """
        Load user-defined functions from file(s).
        Each function has to be wrapped in a Python object using the self.julia.eval command.
        Anything function that takes in numpy arrays must be wrapped in pyfunction([func], PyArray).
        """

        self.julia = julia.Julia(compiled_modules=False)
        print("Loading Julia. This will take ~30 s.")

        if isinstance(self.julia_file, str):
            self.julia.include(self.julia_file)
        else:
            for f in self.julia_file:
                self.julia.include(f)

        # Define functions: set conversion to zero-copy PyArray
        self.j_func.append(self.julia.eval("pyfunction(get_mean, PyArray)"))

    def stop(self):
        print("Julia Analysis broke, avg time per frame: ", np.mean(self.t_per_frame))
        print("JuliaAnalysis got through ", self.frame_number, " frames.")

    def runStep(self):
        t = time.time()
        try:
            obj_id = self.q_in.get(timeout=0.0001)  # List
            self.frame = self.client.getID(
                obj_id[0][str(self.frame_number)]
            )  # Expected np.ndarray
        except Empty:
            pass
        except KeyError as e:  # TODO: Remove reliance on key
            logger.error("Processor: Key error... {0}".format(e))
        except ObjectNotFoundError:
            logger.error("Unavailable from store, dropping frame_number.")

        else:
            self.frame_number += 1
            self.run_julia_analyses()
            self.t_per_frame.append([time.time() - t])

    def run_julia_analyses(self):
        for f in self.j_func:
            self.result_ex = f(np.array(self.frame))

        assert np.isclose(np.mean(self.frame), self.result_ex)
        # print(f'{colorama.Fore.GREEN} Julia: mean intensity of frame {self.frame_number} is {self.result_ex}')

    def export(self):
        t = time.time()
        obj_ids = [self.client.put([self.result_ex, "resultJulia" + self.frame_number])]

        # Automatic generation of variables to export.
        # export_list = [key for key, value in self.__dict__.items() if key.endswith('_ex')]
        # obj_ids = list()
        # for var in export_list:
        #     obj_ids.append(self.client.put(eval(f'self.{var}'), f'{var[:-3]}{curr_frame}'))

        self.q_out.put(obj_ids)
        self.t_per_put.append(time.time() - t)

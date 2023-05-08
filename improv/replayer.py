import time
from typing import Callable, List

from pyarrow.plasma import ObjectID

from nexus.actor import Actor, RunManager
from nexus.store import LMDBStore, LMDBData


class Replayer(Actor):
    def __init__(self, *args, lmdb_path, replay: str, resave=False, **kwargs):
        """
        Class that outputs objects to queues based on a saved previous run.

        :param lmdb_path: path to LMDB folder
        :param replay: named of Actor to replay.
        :param resave: (if using LMDB in this instance) save outputs from this actor as usual (default: False)

        """
        super().__init__(*args, **kwargs)
        self.resave = resave

        self.lmdb = LMDBStore(path=lmdb_path, load=True)
        self.lmdb_values: list = self.get_lmdb_values(replay)
        assert len(self.lmdb_values) > 0

        self.gui_messages: dict = self.get_lmdb_values(
            'GUI', func=lambda x: {lmdbdata.obj[0]: lmdbdata.time for lmdbdata in x}
        )

        self.t_saved_start_run = self.gui_messages['run']  # TODO Add load GUI actions
        self.t_start_run = None

    def get_lmdb_values(self, replay: str, func: Callable = None) -> List[LMDBData]:
        """
        Load saved queue objects from LMDB

        :param replay: named of Actor
        :param func: (optional) Function to apply to objects before returning
        :return:
        """
        # Get all out queue names
        replay = f'q__{replay}'
        keys = [
            key.decode()
            for key in self.lmdb.get_keys()
            if key.startswith(replay.encode())
        ]

        # Get relevant keys, convert to str, and sort. Then convert back to bytes.
        keys = [key.encode() for key in keys]
        lmdb_values = sorted(
            self.lmdb.get(keys, include_metadata=True),
            key=lambda lmdb_value: lmdb_value.time,
        )

        if func is not None:
            return func(lmdb_values)
        return lmdb_values

    def setup(self):
        if self.client.use_hdd and not self.resave:
            self.client.use_hdd = False

        self.move_to_plasma(self.lmdb_values)
        self.put_setup(self.lmdb_values)

    def move_to_plasma(self, lmdb_values):
        """Put objects into current plasma store and update object ID in saved queue."""

        # TODO Make async to enable queue-based fetch system to avoid loading everything at once.
        for lmdbdata in lmdb_values:
            try:
                if len(lmdbdata.obj) == 1 and isinstance(
                    lmdbdata.obj[0], dict
                ):  # Raw frames
                    data = lmdbdata.obj[0]
                    for i, obj_id in data.items():
                        if isinstance(obj_id, ObjectID):
                            actual_obj = self.lmdb.get(obj_id, include_metadata=True)
                            lmdbdata.obj = [
                                {i: self.client.put(actual_obj.obj, actual_obj.name)}
                            ]

                for i, obj in enumerate(lmdbdata.obj):  # List
                    if isinstance(obj, ObjectID):
                        actual_obj = self.lmdb.get(obj, include_metadata=True)
                        lmdbdata.obj[i] = self.client.put(
                            actual_obj.obj, actual_obj.name
                        )

                else:  # Not object ID, do nothing.
                    pass

            except (TypeError, AttributeError):  # Something else.
                pass

    def put_setup(self, lmdb_values):
        """Put all objects created before Run into queue immediately."""
        for lmdb_value in lmdb_values:
            if lmdb_value.time < self.t_saved_start_run:
                getattr(self, lmdb_value.queue).put(lmdb_value.obj)

    def run(self):
        self.t_start_run = time.time()
        with RunManager(
            self.name, self.runner, self.setup, self.q_sig, self.q_comm
        ) as rm:
            print(rm)

    def runner(self):
        """
        Get list of objects and output them to their respective queues based on time delay.

        """
        for lmdb_value in self.lmdb_values:
            if lmdb_value.time >= self.t_saved_start_run:
                t_sleep = (
                    lmdb_value.time + self.t_start_run - self.t_saved_start_run
                ) - time.time()
                if t_sleep > 0:
                    time.sleep(t_sleep)
                getattr(self, lmdb_value.queue).put(lmdb_value.obj)

    #     policy = asyncio.get_event_loop_policy()
    #     policy.set_event_loop(policy.new_event_loop())
    #     self.loop = asyncio.get_event_loop()
    #
    #     self.aqueue = asyncio.Queue()
    #     self.loop.run_until_complete(self.arun())
    #
    # async def arun(self):
    #
    #
    #     funcs_to_run = [self.send_q, self.fetch_lmdb]
    #     async with AsyncRunManager(self.name, funcs_to_run, self.setup, self.q_sig, self.q_comm) as rm:
    #         print(rm)
    #
    # async def send_q(self):
    #
    #     for t in self.times:
    #         now = time.time()
    #         await asyncio.sleep(t - now)
    #         self.q_out.put(list(dict()))

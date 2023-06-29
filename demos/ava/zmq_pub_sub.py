from collections import deque
import logging
import numpy as np
import time
import zmq


# LOGGER = logging.getLogger()

maxlen_data = 3200 # 1 s of data
maxlen_ns = 2000 # 2000 msgs each w/diff n

sub_ip = "10.122.168.184"
sub_port = "5555"

pub_ip = "10.122.161.27"
pub_port = "5555"

data = deque([], maxlen=maxlen_data)
ns = deque([], maxlen_ns)

sub = True


class ZMQPubSub:

    def __init__(self, *args, sub_ip=None, sub_port=None, pub_ip=None, pub_port=None, sub=True, pub=False):


        # Create a zmq Context
        # Return a global Context instance
        # Single-threaded app = single, global Context
        self.context = zmq.Context.instance()

        self.sub = sub
        if self.sub is True:
                
            self.sub_ip = sub_ip
            self.sub_port = sub_port

            print(self.sub_ip, self.sub_port)

        self.pub = pub
        if self.pub is True:

            self.pub_id = pub_ip
            self.sub_port = sub_port


    def subscribe(self):
    # SYNC PUB/SUB w/REQ/REP
    # https://github.com/zeromq/pyzmq/blob/main/examples/pubsub/subscriber.py
    # SUB
        subscriber = self.context.socket(zmq.SUB)
        # '{protocol}://{interface}:{port}'
        print(f"tcp://{self.sub_ip}:{self.sub_port}")
        subscriber.connect(f"tcp://{self.sub_ip}:{self.sub_port}")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        return subscriber


    # PUB
    def publish(self, pub_ip, pub_port):
        publisher = self.context.socket(zmq.PUB)
        # '{protocol}://{interface}:{port}'
        publisher.bind(f"tcp://{pub_ip}:{pub_port}")

        return publisher


# Receive multi-part message from LV
    # """Receive a multipart message as a list of bytes or Frame objects
    # recv_multipart(flags: int = 0, *, copy: Literal[True], track: bool = False) → List[bytes] (default)
    # recv_multipart(flags: int = 0, *, copy: Literal[False], track: bool = False) → List[Frame]
    # recv_multipart(flags: int = 0, *, track: bool = False) → List[bytes]
    # recv_multipart(flags: int = 0, copy: bool = True, track: bool = False) → List[Frame] | List[bytes]

    def run(self):

        if self.sub is True:
            subscriber = self.subscribe()

        if self.pub is True:
            publisher = self.publish()

        while 1:
            if self.sub is True:
                try:
                    msg_recv = subscriber.recv_multipart()
                    # print(msg_recv)
                    # msg_send = f"N read: {msg_recv[-1]}"
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        print("error?")
                        # LOGGER.debug(f"e?")
                        pass

                # data = [:-1], all except last value
                # n = [-1], last value
                # Out Vector and LeftOverAmp
                data.extend(np.int16(msg_recv[:-1]))
                ns.append(int(msg_recv[-1]))

                # ns number of samples/msg
                # N total number of samples up to this point, should = len of data
                N = np.sum(ns)
                D = len(data)

                print(list(data)[:200])

                print(len(data))
                if  N >= 320:

                    print(f"N: {N}")

                    d = list(data)

                    print(f"data: {d}")

                    out, ex = d[:320], d[320:]
                    # save out, save left


                    time.sleep(1)

                    print(f"data: {data} \n \n" \
                        f"ns: {ns} \n \n" \
                        f"N: {np.sum(ns)} \n \n")
                
            # if self.pub is True:




# Close all sockets associated w/context, terminate the context
# If active sockets in other threads, DO NOT call
# context.destroy()

zmqps = ZMQPubSub(sub_ip=sub_ip, sub_port=sub_port, sub=True)
zmqps.run()

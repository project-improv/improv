import time
import os
import h5py
import struct
import numpy as np
import random
import ipaddress
import zmq
import json
from pathlib import Path
from improv.actor import Actor, Spike, RunManager
from queue import Empty

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZMQAcquirer(Actor):

    def __init__(self, *args, ip=None, ports=None, **kwargs):
        super().__init__(*args, **kwargs)
        print("init")
        self.ip = ip
        self.ports = ports
        self.frame_num = 0

        # Sanity check
        # ipaddress.ip_address(self.ip)  # Check if IP is valid.
        # if not 0 <= port <= 65535:
        #     raise ValueError(f'Port {self.port} invalid.')

        # self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.SUB)
        # self.socket.connect(f"tcp://{self.ip}:{self.port}")
        # self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topicfilter)

    def setup(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        for port in self.ports:
            self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
            print('Connected to '+str(self.ip)+':'+str(port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.saveArray = []
        self.save_ind = 0
        self.fullStimmsg = []

        self.tailF = False
        self.stimF = False
        self.frameF = False

        ## TODO: save initial set of frames to data/init_stream.h5

    def run(self):
        '''Triggered at Run
        '''
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
        print('-------------------------------------------- HERE')

        self.imgs = np.array(self.saveArray)
        f = h5py.File('output/sample_stream.h5', 'w', libver='earliest')
        f.create_dataset("default", data=self.imgs)
        f.close()

        np.savetxt('output/stimmed.txt', np.array(self.stimmed))
        np.savetxt('output/tails.txt', np.array(self.tails))
        np.savetxt('output/timing/frametimes.txt', np.array(self.frametimes))
        np.savetxt('output/timing/framesendtimes.txt', np.array(self.framesendtimes), fmt="%s")
        np.savetxt('output/timing/stimsendtimes.txt', np.array(self.stimsendtimes), fmt="%s")
        np.savetxt('output/timing/tailsendtimes.txt', np.array(self.tailsendtimes), fmt="%s")

        print('Acquisition complete, avg time per frame: ', np.mean(self.total_times))
        print('Acquire got through ', self.frame_num, ' frames')
        np.savetxt('output/timing/acquire_frame_time.txt', self.total_times)
        np.savetxt('output/timing/acquire_timestamp.txt', self.timestamp)

        np.save('output/fullstim.npy', self.fullStimmsg)

    def runAcquirer(self):
        print("runAcquirer")
        ''' Main loop. If there're new files, read and put into store.
        '''
        t = time.time()
        #TODO: use poller instead to prevent blocking, include a timeout
        try:
            msg = self.socket.recv(flags=zmq.NOBLOCK)
            # print('msg:', msg)
            msg_parts = [part.strip() for part in msg.split(b': ', 1)]
            # logger.info(msg_parts[0].split(b' ')[:4])
            tag = msg_parts[0].split(b' ')[0]
            # sendtime = msg_parts[0].split(b' ')[1].decode()
            # print(tag)

            if 'stimid' in str(tag): #tag == b'stimid':
                # print(msg)
                if not self.stimF:
                    logger.info('Receiving stimulus information')
                    self.stimF = True
                self.fullStimmsg.append(msg)
                msg_parts = [part.strip() for part in msg.split(b" ")]
                # print('------------')
                # print(msg_parts)
                # print('############')
                # new_dicts = []
                # for z in range(10):
                #     new_dicts.append([j for j in [i.split(b' ') for i in msg.split(b'\n')][z] if j is not b''])
                # print(new_dicts)
                sendtime = msg_parts[2].decode()
                types = str(msg_parts[5].decode()[:-1])
                category = str(msg_parts[3].decode()[:-1])
                # sendtime = new_dicts[0][2].decode()
                # types = 's'
                # category = new_dicts[0][3].decode()
                
                if 's' in str(types) and 'moving' in category:
                    # print(new_dicts[1][1].decode())
                    angle = (float(msg_parts[7].decode()[:-1])) #int(new_dicts[1][1].decode()) #int(msg_parts[7].decode()[:-1])
                    vel = np.abs(float(msg_parts[9].decode()[:-1])) #np.abs(float(new_dicts[8][1].decode())) #np.abs(float(msg_parts[9].decode()[:-1]))
                    freq = float(msg_parts[15].decode()[:-1]) #float(new_dicts[9][1].decode()) #float(msg_parts[15].decode()[:-1])
                    light = 0#int(msg_parts[17].decode()[:-1])
                    dark = 240#int(msg_parts[19].decode()[:-1])
                    # print('stim sendtime ', sendtime)
                    # print('stimulus id: {}, {}'.format(angle, vel))

                    #calibration check:
                    # angle = 270+angle
                    if angle>=360:
                        angle-=360
                    # angle = int(float(angle))

                    stim = 0
                    # stimonOff = 20
                    if np.abs(vel) > 0:
                        stimonOff = 20
                    else:
                        stimonOff = 0

                    # if msg_parts[1] == b'Left':
                    #     stim = 4
                    # elif msg_parts[1] == b'Right':
                    #     stim = 3
                    # elif msg_parts[1] == b'forward':
                    #     stim = 9
                    # elif msg_parts[1] == b'backward':
                    #     stim = 13
                    # elif msg_parts[1] == b'background_stim':
                    #     stimonOff = 0
                    #     print('Stim off')
                    # elif msg_parts[1] == b'Left_Backward':
                    #     stim = 14
                    # elif msg_parts[1] == b'Right_Backward':
                    #     stim = 12
                    # elif msg_parts[1] == b'Left_Forward':
                    #     stim = 16
                    # elif msg_parts[1] == b'Right_Forward':
                    #     stim = 10

                    if 23 > angle >=0:
                        stim = 9
                    elif 360 > angle >= 338:
                        stim = 9
                    elif 113 > angle >= 68:
                        stim = 3
                    elif 203 > angle >= 158:
                        stim = 13
                    elif 293 > angle >= 248:
                        stim = 4

                    elif 68 > angle >= 23:
                        stim = 10
                    elif 158 > angle >= 113:
                        stim = 12
                    elif 248 > angle >= 203:
                        stim = 14
                    elif 338 > angle >= 293:
                        stim = 16
                    else:
                        logger.error('Stimulus unrecognized')
                        print(msg_parts)
                        print(angle)

                    if dark == 30:
                        contrast = 0
                    elif dark == 120:
                        contrast = 1
                    elif dark == 240 and light == 0:
                        contrast = 2
                    elif light == 120:
                        contrast = 3
                    elif light == 210:
                        contrast = 4
                    else:
                        logger.error('Stimulus unrecognized')
                        print(msg_parts)
                        print(angle)

                    if stimonOff:
                        logger.info('Stimulus: {}, angle: {}, speed: {}, freq: {}, frame {}'.format(stim, angle, vel, freq, self.frame_num))
                    

                    self.links['stim_queue'].put({self.frame_num:[stim, stimonOff, angle, vel, freq, contrast]})
                    # for i in np.arange(self.frame_num,self.frame_num+15): # known stimulus duration
                        # self.links['stim_queue'].put({i:[stim, stimonOff]})
                    self.stimmed.append([self.frame_num, stim, angle, vel, freq, contrast, time.time()])
                    self.stimsendtimes.append([sendtime])

            elif tag == b'frame':
                if not self.frameF:
                    logger.info('Receiving frame information')
                    self.frameF = True
                t0 = time.time()
                sendtime =  msg_parts[0].split(b' ')[2].decode()
                # print(sendtime)
                array = np.array(json.loads(msg_parts[1]))[:,32:]  # assuming the following message structure: 'tag: message'
                # print(array.shape)
                # print('frame ', self.frame_num)
                # print('{}'.format(msg_parts[0])) # messsage length: {}. Element sum: {}; time to process: {}'.format(msg_parts[0], len(msg),
                                                                                            # array.sum(), time.time() - t0))
                # output example: b'frame ch0 10:02:01.115 AM 10/11/2019' messsage length: 1049637. Element sum: 48891125; time to process: 0.04192757606506348
                
                # array = np.flip(array, axis=0).copy()
                obj_id = self.client.put(array, 'acq_raw' + str(self.frame_num))
                self.q_out.put([{str(self.frame_num): obj_id}])

                self.saveArray.append(array)
                self.frametimes.append([self.frame_num, time.time()])
                self.framesendtimes.append([sendtime])

                if len(self.saveArray) >= 1000:
                    self.imgs = np.array(self.saveArray)
                    f = h5py.File('output/sample_stream'+str(self.save_ind)+'.h5', 'w', libver='earliest')
                    f.create_dataset("default", data=self.imgs)
                    f.close()
                    self.save_ind += 1
                    self.saveArray = []

                self.frame_num += 1
                self.total_times.append(time.time() - t0)

            elif tag == b'tail':
                if not self.tailF:
                    logger.info('Receiving tail information')
                    self.tailF = True
                t0 = time.time()
                sendtime = msg_parts[0].split(b' ')[1].decode()
                msg = msg_parts[0].split(b' ')[4:]
                # print(msg)
                vlist = []
                for i,m in enumerate(msg):
                    try:
                        v = int(m.decode())
                    except ValueError:
                        v = int(m.decode()[:-1])
                    vlist.append(v)
                self.tails.append(np.array(vlist))
                self.tailsendtimes.append([sendtime])

            else:
                # if len(msg) < 500:
                print('--------frame: ', self.frame_num, ' ### ', msg)
                # else:
                #     print('msg length: {}'.format(len(msg)))

        except zmq.Again as e:
            pass #no messages available
        except Exception as e:
            print('error: {}'.format(e))

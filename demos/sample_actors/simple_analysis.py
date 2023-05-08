import time
import numpy as np
from queue import Empty
import os

from improv.actor import Actor, RunManager
from improv.store import ObjectNotFoundError

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleAnalysis(Actor):
    # TODO: Add additional error handling
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self, param_file=None):
        '''Set custom parameters here
        Can also be done by e.g. loading them in from
        a configuration file. #TODO
        '''
        np.seterr(divide='ignore')

        self.num_stim = 21
        self.frame = 0
        self.stim = {}
        self.stimStart = {}
        self.window = 500  # TODO: make user input, choose scrolling window for Visual
        self.C = None
        self.S = None
        self.Call = None
        self.Cx = None
        self.Cpop = None
        self.coords = None
        self.color = None
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0] * self.window

        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []

    def stop(self):
        print('Analysis broke, avg time per frame: ', np.mean(self.total_times, axis=0))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.puttime))
        print('Analysis broke, avg time per color frame: ', np.mean(self.colortime))
        print('Analysis broke, avg time per stim avg: ', np.mean(self.stimtime))
        print('Analysis got through ', self.frame, ' frames')

        if not os._exists('output'):
            try:
                os.makedirs('output')
            except:
                pass
        if not os._exists('output/timing'):
            try:
                os.makedirs('output/timing')
            except:
                pass
        np.savetxt('output/timing/analysis_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/analysisput_frame_time.txt', np.array(self.puttime))
        np.savetxt(
            'output/timing/analysiscolor_frame_time.txt', np.array(self.colortime)
        )
        np.savetxt('output/timing/analysis_timestamp.txt', np.array(self.timestamp))

    def runStep(self):
        '''Take numpy estimates and frame_number
        Create X and Y for plotting
        '''
        t = time.time()
        ids = None
        try:
            print(self.links)
            sig = self.links['input_stim_queue'].get(timeout=0.0001)
            self.updateStim_start(sig)
        except Empty as e:
            pass  # no change in input stimulus
        try:
            ids = self.q_in.get(timeout=0.0001)
            ids = [id[0] for id in ids]
            if ids is not None and ids[0] == 1:
                print('analysis: missing frame')
                self.total_times.append(time.time() - t)
                self.q_out.put([1])
                raise Empty
            # t = time.time()
            self.frame = ids[-1]
            (self.coordDict, self.image, self.S) = self.client.getList(ids[:-1])
            self.C = self.S
            self.coords = [o['coordinates'] for o in self.coordDict]

            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            self.stimAvg_start()

            self.globalAvg = np.mean(self.estsAvg[:, :8], axis=0)
            self.tune = [self.estsAvg[:, :8], self.globalAvg]

            if self.frame >= self.window:
                window = self.window
            else:
                window = self.frame

            if self.C.shape[1] > 0:
                self.Cpop = np.nanmean(self.C, axis=0)
                self.Cx = np.arange(0, self.Cpop.size) + (self.frame - window)
                self.Call = (
                    self.C
                )  # already a windowed version #[:,self.frame-window:self.frame]

            self.putAnalysis()
            self.timestamp.append([time.time(), self.frame])
            self.total_times.append(time.time() - t)
        except ObjectNotFoundError:
            logger.error('Estimates unavailable from store, droppping')
        except Empty as e:
            pass
        except Exception as e:
            logger.exception('Error in analysis: {}'.format(e))

    def updateStim(self, stim):
        '''Recevied new signal from some Acquirer to change input stimulus
        [possibly other action items here...? Validation?]
        '''
        # stim in format dict frame_num:[n, on/off]
        frame = list(stim.keys())[0]
        whichStim = stim[frame][0]

        # stim is dict with stimID as key and lists for indexing on/off into that stim
        # length is number of frames
        if whichStim not in self.stim.keys():
            self.stim.update({whichStim: {}})
        if abs(stim[frame][1]) > 1:
            if 'on' not in self.stim[whichStim].keys():
                self.stim[whichStim].update({'on': []})
            self.stim[whichStim]['on'].append(frame)
        else:
            if 'off' not in self.stim[whichStim].keys():
                self.stim[whichStim].update({'off': []})
            self.stim[whichStim]['off'].append(frame)

        # also store which stim is active for each frame, up to a recent window
        self.recentStim[frame % self.window] = whichStim

    def updateStim_start(self, stim):
        frame = list(stim.keys())[0]
        whichStim = stim[frame][0]
        if whichStim not in self.stimStart.keys():
            self.stimStart.update({whichStim: []})
        if abs(stim[frame][1]) > 1:
            curStim = 1  # on
        else:
            curStim = 0  # off
        if self.lastOnOff is None:
            self.lastOnOff = curStim
        elif self.lastOnOff == 0 and curStim == 1:  # was off, now on
            self.stimStart[whichStim].append(frame)
            print('Stim ', whichStim, ' started at ', frame)

        self.lastOnOff = curStim

    def putAnalysis(self):
        '''Throw things to DS and put IDs in queue for Visual'''
        t = time.time()
        ids = []
        ids.append(
            [self.client.put(self.Cx, 'Cx' + str(self.frame)), 'Cx' + str(self.frame)]
        )
        ids.append(
            [
                self.client.put(self.Call, 'Call' + str(self.frame)),
                'Call' + str(self.frame),
            ]
        )
        ids.append(
            [
                self.client.put(self.Cpop, 'Cpop' + str(self.frame)),
                'Cpop' + str(self.frame),
            ]
        )
        ids.append(
            [
                self.client.put(self.tune, 'tune' + str(self.frame)),
                'tune' + str(self.frame),
            ]
        )
        ids.append(
            [
                self.client.put(self.color, 'color' + str(self.frame)),
                'color' + str(self.frame),
            ]
        )
        ids.append(
            [
                self.client.put(self.coordDict, 'analys_coords' + str(self.frame)),
                'analys_coords' + str(self.frame),
            ]
        )
        ids.append([self.frame, str(self.frame)])

        self.put(ids, save=[False, False, False, False, False, True, False])

        self.puttime.append(time.time() - t)

    def stimAvg_start(self):
        ests = self.S  # ests = self.C
        ests_num = ests.shape[1]
        t = time.time()
        self.polarAvg = [np.zeros(ests.shape[0])] * 12
        estsAvg = [np.zeros(ests.shape[0])] * self.num_stim
        for s, l in self.stimStart.items():
            l = np.array(l)
            if l.size > 0:
                onInd = np.array(
                    [np.arange(o + 5, o + 35) for o in np.nditer(l)]
                ).flatten()
                onInd = onInd[onInd < ests_num]
                offInd = np.array(
                    [np.arange(o - 20, o - 1) for o in np.nditer(l)]
                ).flatten()  # TODO replace
                offInd = offInd[offInd >= 0]
                offInd = offInd[offInd < ests_num]
                try:
                    if onInd.size > 0:
                        onEst = np.mean(ests[:, onInd], axis=1)
                    else:
                        onEst = np.zeros(ests.shape[0])
                    if offInd.size > 0:
                        offEst = np.mean(ests[:, offInd], axis=1)
                    else:
                        offEst = np.zeros(ests.shape[0])
                    try:
                        estsAvg[int(s)] = onEst  # (onEst / offEst) - 1
                    except FloatingPointError:
                        print('Could not compute on/off: ', onEst, offEst)
                        estsAvg[int(s)] = onEst
                    except ZeroDivisionError:
                        estsAvg[int(s)] = np.zeros(ests.shape[0])
                except FloatingPointError:  # IndexError:
                    logger.error('Index error ')
                    print('int s is ', int(s))
            # else:
            #     estsAvg[int(s)] = np.zeros(ests.shape[0])

        estsAvg = np.array(estsAvg)
        self.polarAvg[2] = estsAvg[9, :]  # np.sum(estsAvg[[9,11,15],:], axis=0)
        self.polarAvg[1] = estsAvg[10, :]
        self.polarAvg[0] = estsAvg[3, :]  # np.sum(estsAvg[[3,5,8],:], axis=0)
        self.polarAvg[7] = estsAvg[12, :]
        self.polarAvg[6] = estsAvg[13, :]  # np.sum(estsAvg[[13,17,18],:], axis=0)
        self.polarAvg[5] = estsAvg[14, :]
        self.polarAvg[4] = estsAvg[4, :]  # np.sum(estsAvg[[4,6,7],:], axis=0)
        self.polarAvg[3] = estsAvg[16, :]

        # for color summation
        self.polarAvg[8] = estsAvg[5, :]
        self.polarAvg[9] = estsAvg[6, :]
        self.polarAvg[10] = estsAvg[7, :]
        self.polarAvg[11] = estsAvg[8, :]

        self.estsAvg = np.abs(np.transpose(np.array(self.polarAvg)))
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.stimtime.append(time.time() - t)

    def stimAvg(self):
        ests = self.S  # ests = self.C
        ests_num = ests.shape[1]
        # S = self.S
        t = time.time()
        self.polarAvg = [np.zeros(ests.shape[0])] * 8
        estsAvg = [np.zeros(ests.shape[0])] * self.num_stim

        if self.runMeanOn is None:
            self.runMeanOn = [np.zeros(ests.shape[0])] * self.num_stim
        if self.runMeanOff is None:
            self.runMeanOff = [np.zeros(ests.shape[0])] * self.num_stim
        if self.runMean is None:
            self.runMean = [np.zeros(ests.shape[0])] * self.num_stim

        if self.frame > 0:  # self.window: #recompute entire mean
            for s, l in self.stim.items():
                if 'on' in l.keys() and 'off' in l.keys():
                    onInd = np.array(l['on'])
                    onInd = onInd[onInd < ests_num]
                    offInd = np.array(l['off'])
                    offInd = offInd[offInd < ests_num]
                    try:
                        on = np.mean(ests[:, onInd], axis=1)
                        off = np.mean(ests[:, offInd], axis=1)
                        try:
                            estsAvg[int(s)] = (on / off) - 1
                        except FloatingPointError:
                            print('Could not compute on/off: ', on, off)
                            estsAvg[int(s)] = on
                    except IndexError:
                        logger.error('Index error ')
                        print('int s is ', int(s))
                else:
                    estsAvg[int(s)] = np.zeros(ests.shape[0])
        else:
            # keep running mean as well as recalc mean for possible updates
            # ests only contains self.window number of most recent frames
            # running mean of last newest frame, recalc mean of all more recent frames
            for s, l in self.stim.items():
                print(s, l)
                if 'on' in l.keys() and 'off' in l.keys():
                    onInd = np.array(l['on'])
                    offInd = np.array(l['off'])
                    # print('onInd ', onInd)
                    # print('offInd ', offInd)
                    try:
                        if self.frame == onInd[-1]:
                            self.runMeanOn[int(s)] += np.mean(
                                ests[:, onInd[-1]], axis=1
                            )
                        elif self.frame == offInd[-1]:
                            self.runMeanOff[int(s)] += np.mean(
                                ests[:, offInd[-1]], axis=1
                            )
                        on = np.mean(ests[:, onInd[:-1]], axis=1)
                        off = np.mean(ests[:, offInd[:-1]], axis=1)
                        try:
                            estsAvg[int(s)] = (on / off) - 1
                        except FloatingPointError:
                            estsAvg[int(s)] = on
                    except IndexError:
                        pass
                else:
                    estsAvg[int(s)] = np.zeros(ests.shape[0])

        estsAvg = np.array(estsAvg)
        self.polarAvg[2] = estsAvg[9, :]  # np.sum(estsAvg[[9,11,15],:], axis=0)
        self.polarAvg[1] = estsAvg[10, :]
        self.polarAvg[0] = estsAvg[3, :]  # np.sum(estsAvg[[3,5,8],:], axis=0)
        self.polarAvg[7] = estsAvg[12, :]
        self.polarAvg[6] = estsAvg[13, :]  # np.sum(estsAvg[[13,17,18],:], axis=0)
        self.polarAvg[5] = estsAvg[14, :]
        self.polarAvg[4] = estsAvg[4, :]  # np.sum(estsAvg[[4,6,7],:], axis=0)
        self.polarAvg[3] = estsAvg[16, :]

        # for color summation
        self.polarAvg[8] = estsAvg[5, :]
        self.polarAvg[9] = estsAvg[6, :]
        self.polarAvg[10] = estsAvg[7, :]
        self.polarAvg[11] = estsAvg[8, :]

        self.estsAvg = np.abs(np.transpose(np.array(self.polarAvg)))
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        # self.estsAvg = np.clip(self.estsAvg*4, 0, 4)
        self.stimtime.append(time.time() - t)

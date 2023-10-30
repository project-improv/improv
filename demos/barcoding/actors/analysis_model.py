from improv.actor import Actor, Signal, RunManager
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2
import colorsys
import scipy
import pickle
from scipy import stats
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelAnalysis(Actor):
    def __init__(self, *args, input_stim_num = 8):
        super().__init__(*args)
        N = 5  # guess XX neurons initially
        dh = 4
        dk = 8
        w = np.zeros((N, N))
        h = np.zeros((N, dh))
        k = np.zeros((N, dk))
        b = np.zeros(N)
        self.theta = np.concatenate((w, h, b, k), axis=None).flatten()
        self.num_stim = input_stim_num
        self.p = {
            "numNeurons": N,
            "hist_dim": dh,
            "numSamples": 1,
            "dt": 0.5,
            "stim_dim": self.num_stim,
        }  # TODO: from config file..

    def setup(self, param_file=None):
        """ """
        np.seterr(divide="ignore")
        logger.info("Running setup for " + self.name)
        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.ests = np.zeros(
            (1, self.num_stim, 2)
        )  # number of neurons, number of stim, on and baseline
        self.ests_avg = np.zeros(
            (1, self.num_stim, 2)
        )
        self.baseline_coefficient = np.zeros((1, self.num_stim, 3))
        self.baseline_const_coefficient = np.zeros((1, self.num_stim, 2))
        self.trial_avg_ptvalue = np.zeros((1, self.num_stim, 2))
        self.trial_avg_max_ptvalue = np.zeros((1, self.num_stim, 2))
        self.trial_slope_ptvalue = np.zeros((1, self.num_stim, 2))
        self.trial_avg_record = {}
        self.trial_avg_max_record = {}
        self.trial_slope_record = {}
        self.onstim_counter = np.ones((self.num_stim, 2))
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
        self.currStimID = np.zeros((8, 1000000))  # FIXME
        self.currStim = -10
        self.allStims = {}
        self.estsAvg = None
        self.final_barcode = np.zeros((1, self.num_stim))
        self.onstim_counter = np.zeros((self.num_stim, 2))
        self.counter = np.ones((self.num_stim, 2))
        self.estRecord = {}
        self.baseline_record = None
        self.baseline_const_record = None
        self.barcode_category = {}
        self.trial_count = np.zeros((self.num_stim,))
        for i in range(self.num_stim):
            self.estRecord[i] = None
            self.trial_avg_record[i] = None
            self.trial_avg_max_record[i] = None
            self.trial_slope_record[i] = None
        self.mean_barcode_both = np.zeros((1, 8))
        self.mean_barcode_slope = np.zeros((1, 8))
        self.mean_barcode_avg = np.zeros((1, 8))
        self.both_mean_on = np.zeros(8)
        self.both_mean = {}
        self.fitting_barcode = np.zeros((1, 8))
        self.fitting = {}
        self.fitting_on = np.zeros(8)
        self.rest_barcode = np.zeros((1, 8))
        self.rest = {}
        self.rest_on = np.zeros(8)


    def run(self):
        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []
        self.LL = []
        self.fit_times = []

        #self.links = {}
        self.links['q_sig'] = self.q_sig
        self.links['q_comm'] = self.q_comm
        self.actions = {}
        self.actions['run'] = self.runStep
        self.actions['setup'] = self.setup
        
        with RunManager(
            self.name, self.actions, self.links
        ) as rm:
            logger.info(rm)

        print("Analysis broke, avg time per frame: ", np.mean(self.total_times, axis=0))
        print("Analysis broke, avg time per put analysis: ", np.mean(self.puttime))
        print("Analysis broke, avg time per put analysis: ", np.mean(self.fit_times))
        print("Analysis broke, avg time per color frame: ", np.mean(self.colortime))
        print("Analysis broke, avg time per stim avg: ", np.mean(self.stimtime))
        print("Analysis got through ", self.frame, " frames")

        N = self.p["numNeurons"]
        np.savetxt("output/model_weights.txt", self.theta[: N * N].reshape((N, N)))

        np.savetxt("output/timing/analysis_frame_time.txt", np.array(self.total_times))
        np.savetxt("output/timing/analysis_timestamp.txt", np.array(self.timestamp))
        np.savetxt("output/analysis_estsAvg.txt", np.array(self.estsAvg))
        np.savetxt("output/analysis_proc_S_full3.txt", self.S)
        np.savetxt("output/analysis_LL.txt", np.array(self.LL))

        np.savetxt("output/used_stims.txt", self.currStimID)

    def runStep(self):
        """Take numpy estimates and frame_number
        Create X and Y for plotting
        """
        t = time.time()
        ids = None

        try:
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0] == 1:
                print("analysis: missing frame")
                self.total_times.append(time.time() - t)
                self.q_out.put([1])
                raise Empty
            # t = time.time()
            self.frame = ids[-1]
            (self.coordDict, self.image, self.S) = self.client.getList(ids[:-1])
            #logger.info("what is the coordDict? {0}".format(self.coordDict))
            self.C = self.S
            #logger.info("What is the size of input frame? {0}".format(np.shape(self.C)))
            self.coords = [o["coordinates"] for o in self.coordDict]

            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try:
                sig = self.links["input_stim_queue"].get(timeout=0.0001)
                # logger.info("what's in the stim queue sig? {0}".format(sig))
                self.updateStim_start(sig)
            except Empty as e:
                pass  # no change in input stimulus
            self.stimAvg_start()


            # N  = self.p['numNeurons']
            # dh = self.p['hist_dim']
            # ds = self.p['stim_dim']
            # k = np.square(self.theta[N*(N+dh+1):].reshape((N, ds))*1e6)
            # self.globalAvg = np.mean(k, axis=0)
            # self.tune_k = [k, self.globalAvg]

            # Compute coloring of neurons for processed frame
            # Also rotate and stack as needed for plotting
            # TODO: move to viz, but we don't need to compute this 30 times/sec
            self.color = self.plotColorFrame()

            if self.frame >= self.window:
                window = self.window
                self.Cx = np.arange(self.frame - window, self.frame)
            else:
                window = self.frame
                self.Cx = np.arange(0, self.frame)

            if self.C.shape[1] > 0:
                self.Cpop = np.nanmean(self.C, axis=0)
                self.Call = (
                    self.C
                )  # already a windowed version #[:,self.frame-window:self.frame]

            self.putAnalysis()
            self.timestamp.append([time.time(), self.frame])
            self.total_times.append(time.time() - t)

        except ObjectNotFoundError:
            logger.error("Estimates unavailable from store, droppping")
        except Empty as e:
            pass
        except Exception as e:
            logger.exception("Error in analysis: {}".format(e))


    def updateStim_start(self, stim):
        """Rearrange the info about stimulus into
        cardinal directions and frame <--> stim correspondence.

        self.stimStart is the frame where the stimulus started.
        """
        # print('got stim ', stim)
        # get frame number and stimID
        frame = list(stim.keys())[0]
        # print('got frame ', frame)
        whichStim = stim[frame][0]
        # convert stimID into 8 cardinal directions
        stimID = self.IDstim(int(whichStim))
        # assuming we have one of those 8 stimuli
        if stimID != -10:
            if stimID not in self.allStims.keys():
                self.allStims.update({stimID: []})

                # account for stimuli we haven't yet seen
                # if stimID not in self.stimStart.keys():
                #     self.stimStart.update({stimID:None})

            # determine if this is a new stimulus trial
            if abs(stim[frame][1]) > 1:
                curStim = 1  # on
                self.allStims[stimID].append(frame)
                for i in range(10):
                    self.allStims[stimID].append(frame + i + 1)
            else:
                curStim = 0  # off
            logger.warning("what is frame here?:{0}".format(frame))
            
            # paradigm for these trials is for each stim: [off, on, off]
            if self.lastOnOff is None:
                self.lastOnOff = curStim
                logger.info("current lastonoff is none")
            elif self.lastOnOff == 0 and curStim == 1:  # was off, now on
                # select this frame as the starting point of the new trial
                # and stimulus has started to be shown
                # All other computations will reference this point
                self.stimStart = frame
                self.currentStim = stimID
                if stimID < 8:
                    self.currStimID[stimID, frame] = 1
                # NOTE: this overwrites historical info on past trials
                logger.info("Stim {} started at {}".format(stimID, frame))
            else:
                self.currStimID[:, frame] = np.zeros(8)
            self.lastOnOff = curStim

    def putAnalysis(self):
        """Throw things to DS and put IDs in queue for Visual"""
        t = time.time()
        ids = []
        stim = [self.lastOnOff, self.currStim]
        N = self.p["numNeurons"]
        w = self.theta[: N * N].reshape((N, N))
        barcode_cat = [self.barcode_category]
        ids.append(self.client.put(self.Cx, "Cx" + str(self.frame)))
        ids.append(self.client.put(self.Call, "Call" + str(self.frame)))
        ids.append(self.client.put(self.Cpop, "Cpop" + str(self.frame)))
        ids.append(self.client.put(self.final_barcode, "barcode" + str(self.frame)))
        ids.append(self.client.put(barcode_cat, "barcode_category" + str(self.frame)))
        ids.append(self.client.put(self.color, "color" + str(self.frame)))
        ids.append(self.client.put(self.coordDict, "analys_coords" + str(self.frame)))
        ids.append(self.client.put(self.allStims, "stim" + str(self.frame)))
        ids.append(self.client.put(self.image, "image" + str(self.frame)))
        ids.append(self.frame)

        self.q_out.put(ids)
        self.puttime.append(time.time() - t)


    def fit_line(self, frame_start, stim_ests, stimID):
        #logger.info("input stim_ests: {0}, {1}".format(stim_ests, np.shape(stim_ests)))
        response = stim_ests.T
        duration = np.shape(response)[0]
        regressor = np.linspace(frame_start, frame_start + duration - 1, np.shape(response)[0])
        #logger.info("input regressor and response shape for line fitting:{0}, {1}, {2}".format(regressor, np.shape(regressor), np.shape(response)))
        coefficients = np.polyfit(regressor, response, deg=1).T
        #logger.info("coefficnets shape for line fitting:{0}, {1}".format(np.shape(coefficients), coefficients))
        res = stim_ests - [(regressor * coefficients[i, 0] + coefficients[i,1]) for i in range(np.shape(stim_ests)[0])]
        std = np.std(res, axis = 1)
        #logger.info("std shape for line fitting:{0}, {1}".format(np.shape(std), std))
        self.baseline_coefficient[:, self.currentStim, 0] = coefficients[:,0]
        self.baseline_coefficient[:, self.currentStim, 1] = coefficients[:,1]
        self.baseline_coefficient[:, self.currentStim, 2] = std

    def fit_line_const(self, frame_start, stim_ests, stimID):
        #logger.info("input stim_ests: {0}, {1}".format(stim_ests, np.shape(stim_ests)))
        response = stim_ests.T
        duration = np.shape(response)[0]
        regressor = np.linspace(frame_start, frame_start + duration - 1, np.shape(response)[0])
        #logger.info("input regressor and response shape for line fitting:{0}, {1}, {2}".format(response, np.shape(regressor), np.shape(response)))
        coefficients = np.polyfit(regressor, response, deg=0).T
        #logger.info("coefficnets shape for line fitting:{0}, {1}".format(np.shape(coefficients), coefficients))
        res = stim_ests - (coefficients[:,0])[:, np.newaxis]
        std = np.std(res, axis = 1)
        #logger.info("std shape for line fitting:{0}, {1}".format(np.shape(std), std))
        self.baseline_const_coefficient[:, self.currentStim, 0] = coefficients[:,0]
        self.baseline_const_coefficient[:, self.currentStim, 1] = std
    
    def trial_avg_t_p(self, baseline_part, onstim_part, stimID):
        assert(self.currentStim == stimID)
        baseline_mean = np.mean(baseline_part, axis = 1)
        onstim_mean = np.mean(onstim_part, axis = 1)
        onstim_max = np.mean(onstim_part, axis = 1)
        rvs = onstim_mean - baseline_mean
        rvs_max = onstim_max - baseline_mean
        if self.trial_avg_record[stimID] is None:
            self.trial_avg_record[stimID] = rvs[:, np.newaxis]
        else:
            self.trial_avg_record[stimID] = np.concatenate((self.trial_avg_record[stimID], rvs[:, np.newaxis]), axis = 1)
            #logger.info("test avg_record: {0}".format(self.trial_avg_record[stimID]))
        if self.trial_avg_max_record[stimID] is None:
            self.trial_avg_max_record[stimID] = rvs_max[:, np.newaxis]
        else:
            self.trial_avg_max_record[stimID] = np.concatenate((self.trial_avg_max_record[stimID], rvs_max[:, np.newaxis]), axis = 1)
            #logger.info("test avg_max_record: {0}".format(self.trial_avg_max_record[stimID]))
        
        invalid_neuron_mean = np.where(np.mean(self.trial_avg_record[stimID], axis = 1) < 0)
        #print("the shape of rvs: ", np.shape(rvs))
        stat_res = stats.ttest_1samp(self.trial_avg_record[stimID], popmean=0, axis = 1)
        t_stat_mean = np.array(stat_res[0])
        p_value_mean = np.array(stat_res[1])
        p_value_mean[invalid_neuron_mean] = -1
        self.trial_avg_ptvalue[:, stimID, 0] = np.copy(t_stat_mean)
        self.trial_avg_ptvalue[:, stimID, 1] = np.copy(p_value_mean)

        invalid_neuron_max = np.where(np.mean(self.trial_avg_max_record[stimID], axis = 1) < 0)
        #print("the shape of rvs: ", np.shape(rvs))
        stat_res = stats.ttest_1samp(self.trial_avg_max_record[stimID], popmean=0, axis = 1)
        t_stat_max = np.array(stat_res[0])
        p_value_max = np.array(stat_res[1])
        p_value_max[invalid_neuron_max] = -1
        self.trial_avg_max_ptvalue[:, stimID, 0] = np.copy(t_stat_max)
        self.trial_avg_max_ptvalue[:, stimID, 1] = np.copy(p_value_max)

        #logger.info("print out all the record: {0}, \n {1}".format(self.trial_avg_max_record, self.trial_avg_record))
   
   
    def trial_slope_t_p(self, onstim_part, stimID):
        regressor = np.linspace(0, np.shape(onstim_part)[1], np.shape(onstim_part)[1])
        #print("input regressor and response shape for line fitting: ", np.shape(regressor), np.shape(response))
        coefficients = np.polyfit(regressor, onstim_part.T, deg=1).T
        slope = coefficients[:,0]
        if self.trial_slope_record[stimID] is None:
            self.trial_slope_record[stimID] = slope[:, np.newaxis]
        else:
            self.trial_slope_record[stimID] = np.concatenate((self.trial_slope_record[stimID], slope[:, np.newaxis]), axis = 1)
        #logger.info("test again, {0}".format(self.trial_slope_record))
        res = stats.ttest_1samp(self.trial_slope_record[stimID], popmean=0, axis = 1)
        t_value = np.copy(res[0])
        p_value = np.copy(res[1])
        invalid_index = np.where(np.mean(self.trial_slope_record[stimID], axis = 1) < 0)
        p_value[invalid_index] = -1
        self.trial_slope_ptvalue[:, stimID, 0] = np.copy(t_value)
        self.trial_slope_ptvalue[:, stimID, 1] = np.copy(p_value)

    def baselines_methods(self, i, j):
        if (self.estsAvg_fitline[i, j] + self.estsAvg_avg[i, j]) > 0.05:
            self.fitting_barcode[i, j] = 1
            self.fitting_on[j] += 1
            if self.fitting_on[j] == 1:
                self.fitting[j] = []
                self.fitting[j].append(i)
            else:
                self.fitting[j].append(i)
        if (self.fitting_barcode[i, j] > 0):
            self.final_barcode[i, j] = 1
        else:
            self.final_barcode[i, j] = 0

    def all_methods(self, i, j):
        # Save all the index that are at least three tests said yes
        if ((self.estsAvg_fitline[i, j] + self.estsAvg_avg[i, j]) > 0) and ((self.trial_slope_ptvalue[i, j, 1] > 0) and (self.trial_slope_ptvalue[i, j, 1] < 0.2)) and (((self.trial_avg_max_ptvalue[i, j, 1] > 0) and (self.trial_avg_max_ptvalue[i, j, 1] < 0.2)) or ((self.trial_avg_ptvalue[i, j, 1] > 0) and (self.trial_avg_ptvalue[i, j, 1] < 0.2))):
            self.mean_barcode_both[i, j] = 1
            self.both_mean_on[j] += 1
            if self.both_mean_on[j] == 1:
                self.both_mean[j] = []
                self.both_mean[j].append(i)
            else:
                self.both_mean[j].append(i)
        elif ((self.trial_slope_ptvalue[i, j, 1] > 0) and (self.trial_slope_ptvalue[i, j, 1] < 0.2)):
            self.mean_barcode_slope[i, j] = 1
        elif (((self.trial_avg_max_ptvalue[i, j, 1] > 0) and (self.trial_avg_max_ptvalue[i, j, 1] < 0.2)) or ((self.trial_avg_ptvalue[i, j, 1] > 0) and (self.trial_avg_ptvalue[i, j, 1] < 0.2))):
            self.mean_barcode_avg[i, j] = 1
        if (self.estsAvg_fitline[i, j] + self.estsAvg_avg[i, j]) > 0.05:
            self.fitting_barcode[i, j] = 1
            self.fitting_on[j] += 1
            if self.fitting_on[j] == 1:
                self.fitting[j] = []
                self.fitting[j].append(i)
            else:
                self.fitting[j].append(i)
        if ((self.mean_barcode_slope[i, j] == 1 and (self.trial_avg_ptvalue[i, j, 1] + self.trial_avg_max_ptvalue[i, j, 1] > 0)) or (self.mean_barcode_avg[i, j] == 1 and self.trial_avg_ptvalue[i, j, 1] > 0)):
            if (self.fitting_barcode[i, j] == 1 and ((((self.trial_slope_ptvalue[i, j, 0] + self.trial_avg_max_ptvalue[i, j, 0]) / 2) > 2.92) or (((self.trial_avg_ptvalue[i, j, 0] + self.trial_slope_ptvalue[i, j, 0]) / 2) > 2.92))):
                self.rest_barcode[i, j] = 1
                self.rest_on[j] += 1
                if self.rest_on[j] == 1:
                    self.rest[j] = []
                    self.rest[j].append(i)
                else:
                    self.rest[j].append(i)
                
        if (self.mean_barcode_both[i, j] + self.rest_barcode[i, j] > 0):
            self.final_barcode[i, j] = 1
        else:
            self.final_barcode[i, j] = 0

    def stimAvg_start(self):
        t = time.time()

        ests = self.C
        num_neurons = np.shape(self.ests)[0]
        if self.ests.shape[0] < ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.baseline_coefficient = np.pad(self.baseline_coefficient, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.baseline_const_coefficient = np.pad(self.baseline_const_coefficient, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.trial_avg_ptvalue = np.pad(self.trial_avg_ptvalue, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.trial_avg_max_ptvalue = np.pad(self.trial_avg_max_ptvalue, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.trial_slope_ptvalue = np.pad(self.trial_slope_ptvalue, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.final_barcode = np.pad(self.final_barcode, ((0, diff), (0, 0)), mode="constant")
            self.ests_avg = np.pad(self.ests_avg, ((0, diff), (0, 0), (0, 0)), mode="constant")
            self.mean_barcode_both = np.pad(self.mean_barcode_both, ((0, diff), (0, 0)), mode="constant")
            self.mean_barcode_slope = np.pad(self.mean_barcode_slope, ((0, diff), (0, 0)), mode="constant")
            self.fitting_barcode = np.pad(self.fitting_barcode, ((0, diff), (0, 0)), mode="constant")
            self.rest_barcode = np.pad(self.rest_barcode, ((0, diff), (0, 0)), mode="constant")
            self.final_barcode = np.pad(self.final_barcode, ((0, diff), (0, 0)), mode="constant")

        if self.currentStim is not None:   
            if self.frame == self.stimStart:
                self.baseline_record = np.copy(ests[:, self.frame - 10 : self.frame])
                # baseline before the stimulus onset
            
            elif self.frame in range(self.stimStart + 1, self.frame + 26):
                if self.frame in range(self.stimStart + 1, self.stimStart + 5):
                    self.baseline_record = np.concatenate((self.baseline_record, np.expand_dims(ests[:, self.frame - 1], axis = 1)), axis = 1)
                elif self.frame in range(self.stimStart + 5, self.stimStart + 19):
                    if self.frame == self.stimStart + 5:
                        self.onstim_counter = np.ones((self.num_stim, 2))
                        self.fit_line(self.stimStart - 9, self.baseline_record, self.currentStim)
                        self.trial_avg = np.ones((self.num_stim, 2))

                    frame_baseline_avg = self.baseline_coefficient[:, self.currentStim, 0] + (1.8 * self.baseline_coefficient[:, self.currentStim, 1])
                    frame_baseline_fit = self.frame * self.baseline_coefficient[:, self.currentStim, 0] + self.baseline_coefficient[:, self.currentStim, 1] + (1.8 * self.baseline_coefficient[:, self.currentStim, 2])
                    #logger.info("what is the baseline here?{0}, {1}".format(np.shape(frame_baseline), frame_baseline))
                    self.ests[:, self.currentStim, 1] = (self.ests[:, self.currentStim, 1] * self.onstim_counter[self.currentStim, 1] + frame_baseline_fit) / (self.onstim_counter[self.currentStim, 1] + 1)
                    self.ests_avg[:, self.currentStim, 1] = (self.ests_avg[:, self.currentStim, 1] * self.onstim_counter[self.currentStim, 1] + frame_baseline_avg) / (self.onstim_counter[self.currentStim, 1] + 1)
                    self.onstim_counter[self.currentStim, 1] += 1
                    self.ests[:, self.currentStim, 0] = (self.onstim_counter[self.currentStim, 0] * self.ests[:, self.currentStim, 0] + ests[:, self.frame - 1]) / (self.onstim_counter[self.currentStim, 0] + 1)
                    self.ests_avg[:, self.currentStim, 0] = (self.onstim_counter[self.currentStim, 0] * self.ests_avg[:, self.currentStim, 0] + ests[:, self.frame - 1]) / (self.onstim_counter[self.currentStim, 0] + 1)
                    self.onstim_counter[self.currentStim, 0] += 1

                    if self.frame == self.stimStart + 18:
                        self.trial_count[self.currentStim] += 1
                        baseline = np.copy(ests[:, self.stimStart - 10 : self.stimStart + 5])
                        ests = np.copy(ests[:, self.stimStart +5 : self.frame])
                        self.trial_avg_t_p(baseline, ests, self.currentStim)
                        self.trial_slope_t_p(ests, self.currentStim)
                        #logger.info("testttttttt!!!{0}, \n {1}, \n{2}".format(self.trial_slope_ptvalue, self.trial_avg_ptvalue, self.trial_avg_max_ptvalue))

        self.estsAvg_fitline = np.squeeze((self.ests[:, :, 0] - self.ests[:, :, 1]))
        self.estsAvg_fitline = np.where(np.isnan(self.estsAvg_fitline), 0, self.estsAvg_fitline)
        self.estsAvg_fitline[self.estsAvg_fitline == np.inf] = 0
        self.barcode_fitline = np.where(self.estsAvg_fitline > 0, 1, 0)

        self.estsAvg_avg = np.squeeze((self.ests_avg[:, :, 0] - self.ests_avg[:, :, 1]))
        self.estsAvg_avg = np.where(np.isnan(self.estsAvg_avg), 0, self.estsAvg_avg)
        self.estsAvg_avg[self.estsAvg_avg == np.inf] = 0
        self.barcode_avg = np.where(self.estsAvg_avg > 0, 1, 0)

        if self.currentStim is not None:        
            # general of getting index and corresponding barcode

            for i in range(num_neurons):
                #logger.info("well.....{0}, \n {1}".format(self.trial_count, self.trial_count[self.currentStim]))
                if (self.trial_count[self.currentStim]< 2):
                    self.baselines_methods(i, self.currentStim)
                else:
                    self.all_methods(i, self.currentStim)
                
        category = np.unique(self.final_barcode, axis = 0)
        index_record = {}
        barcode_bytes_record = {}
        for i in range(len(category)):
            cl = category[i]
            bytestr = np.array2string(cl)
            index_record[bytestr] = []
            barcode_bytes_record[bytestr] = np.copy(cl)
        #print(barcode_bytes_record)
        #print(len(category))

        for neuron in range(num_neurons):
            current_barcode = self.final_barcode[neuron]
            index_record[np.array2string(current_barcode)].append(neuron)
            
        self.barcode_category['index_record'] = index_record
        self.barcode_category['bytes_record'] = barcode_bytes_record

        # count_record = []
        # for key in index_record.keys():
        #     current_number = len(index_record[key])
        #     count_record.append(current_number)
        #     logger.info("{0}, : {1}".format(barcode_bytes_record[key], current_number))

        self.stimtime.append(time.time() - t)

    def plotColorFrame(self):
        """Computes colored nicer background+components frame"""
        t = time.time()
        image = self.image
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[..., 3] = 255
        # TODO: don't stack image each time?
        if self.coords is not None:
            for i, c in enumerate(self.coords):
                # c = np.array(c)
                # logger.info("plzzzzzzzz, {0}, \n {1}".format(c, self.coords))
                ind = c[~np.isnan(c).any(axis=1)].astype(int)
                # TODO: Compute all colors simultaneously! then index in...
                cv2.fillConvexPoly(
                    color, ind, self._tuningColor(i, color[ind[:, 1], ind[:, 0]])
                )

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly.

        self.colortime.append(time.time() - t)
        #logger.info("ok let's see what's the color {0}, {1}".format(np.shape(color), color))
        return color

    def _tuningColor(self, ind, inten):
        """ind identifies the neuron by number"""
        ests = self.estsAvg_fitline
        # ests = self.tune_k[0]
        if ests[ind] is not None:
            try:
                return self.manual_Color_Sum(ests[ind])
            except ValueError:
                return (255, 255, 255, 0)
            except Exception:
                print("inten is ", inten)
                print("ests[i] is ", ests[ind])
        else:
            return (255, 255, 255, 50)

    def manual_Color_Sum(self, x):
        """x should be length 12 array for coloring
        or, for k coloring, length 8
        Using specific coloring scheme from Naumann lab
        """
        if x.shape[0] == 8:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 1, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                ]
            )
        elif x.shape[0] == 12:
            mat_weight = np.array(
                [
                    [1, 0.25, 0],
                    [0.75, 1, 0],
                    [0, 2, 0],
                    [0, 0.75, 1],
                    [0, 0.25, 1],
                    [0.25, 0, 1.0],
                    [1, 0, 1],
                    [1, 0, 0.25],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            )
        else:
            print("Wrong shape for this coloring function")
            return (255, 255, 255, 10)

        color = x @ mat_weight

        blend = 0.8
        thresh = 0.2
        thresh_max = blend * np.max(color)

        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)

        if color.any() and np.linalg.norm(color - np.ones(3)) > 0.35:
            color *= 255
            return (color[0], color[1], color[2], 255)
        else:
            return (255, 255, 255, 10)

    def IDstim(self, s):
        """Function to convert stim ID from Naumann lab experiment into
        the 8 cardinal directions they correspond to.
        """
        stim = -10
        if s == 1: # right left
            stim = 0
        elif s == 2: # left right
            stim = 1
        elif s == 3: # right right
            stim = 5 
        elif s == 4: # left left
            stim = 2
        elif s == 5: # right x
            stim = 6 
        elif s == 6: # left x
            stim = 4 
        elif s == 7: # x left
            stim = 3 
        elif s == 8: # x right
            stim = 7 

        return stim

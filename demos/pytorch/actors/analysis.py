
class PyTorchAnalysis(Actor):
    '''
    '''

    def __init__(self, *args, model_path=None):
        super().__init__(*args)
        logger.info(model_path)
        if model_path is None:
            # logger.error("Must specify a pre-trained model path.")
            logger.error("Must specify a JIT-compiled pre-trained model path.")
        else:
            self.model_path = model_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup(self, message):
        ''' Initialize pre-trained model
        '''
        logger.info('Loading model for ' + self.name)
        self.done = False

        t = time.time()

        self.model= torch.jit.load(self.model_path).to(self.device)
        
        self.load_model_time = time.time() - t
        print(message, self.load_model_time*1000.0)

        return self.model

    def run(self):
        '''
        '''
        self.inference_time = []
        self.put_out_time = []

        with RunManager(self.name, self.runAnalysis, self.setup, self.q_sig, self.q_comm) as rm:
            print(rm)
        
        print('Analysis broke, avg time per image: ', np.mean(self.total_times, axis=0))
        print('Analysis got through ', self.image_number, ' images')

        np.savetxt('output/timing/inference_time.txt', np.array(self.inference_time))
        np.savetxt('output/timing/put_out_time.txt', np.array(self.put_out_time))

    def runAnalysis(self):
       ''' Run analysis, i.e., inference. Runs once per image.
            Output is a location in the DS to continually
            place the model output and classification/prediction, with ref number that
            corresponds to the frame number (TODO)
        '''
        in = self._checkProcImages()

        if img is not None:
            t = time.time()
            self.done = False
            try:
                self.in = self.client.getID([str(self.img_num)])
                t1 = time.time()
                self.get_img_out.appennd(t1 - t)
                self.img = self._processImage(self.img, self.img_num)
                t2 = time.time()
                self.proc_img_time.append([t1 - t2])                
                self.timestamp.append([time.time(), self.img_num])
            # Insert exceptions here...ERROR HANDLING, SEE ANNE'S ACTORS - from 1p demo
            except ObjectNotFoundError:
                logger.error('Processor: Image {} unavailable from store, dropping'.format(self.img_num))
                self.dropped_img.append(self.img_num)
                self.q_out.put([1])
            except KeyError as e:
                logger.error('Processor: Key error... {0}'.format(e))
                # Proceed at all costs
                self.dropped_img.append(self.img_num)
            except Exception as e:
                logger.error('Processor error: {}: {} during image number {}'.format(type(e).__name__,
                                                                                            e, self.frame_number))
                print(traceback.format_exc())
                self.dropped_img.append(self.img_num)
            self.img_num += 1
            self.total_times.append(time.time()-t)
        else:
            pass

    def _checkProcImages(self):
        ''' Check to see if we have processed images for analysis (should be tensors (1, 3, 224, 224)) - check instance and size? TODO?
        '''
        # t = time.time()
        try:
            # Timeout = 0 ms
            res = self.q_in.get(timeout=0)
            if torch.is_tensor(res):
                return res
            else:
                logger.info('Input for analysis is not a tensor')
        #TODO: additional error handling
        except Empty:
            logger.info('No processed images, i.e., tensors for analysis')
            return None
        # self.get_in_time = time.time() - t

def classifyImage(self, classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')):
    '''
    '''
    _, predicted = torch.max(self.output.data, 1)


        total += labels.size(0)
        correct += (predicted == labels).sum().item()
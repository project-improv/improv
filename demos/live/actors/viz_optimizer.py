import numpy as np
# from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import multivariate_normal, norm
import matplotlib.pylab as plt

class Optimizer():
    def __init__(self, gamma, var, nu, eta, x_star):
        self.gamma = gamma
        self.variance = var
        self.nu = nu
        self.eta = eta
        self.x_star = x_star        

        self.d = self.x_star.shape[1]

        self.f = None
        self.sigma = None       ## Note: this is actually sigma squared
        self.X_t = None
        self.K_t = None
        self.k_star = None
        self.y = None
        self.A = None

        self.t = 0

    def kernel(self, x, x_j):
        ## x shape: (T, d) (# tests, # dimensions)
        K = np.zeros((x.shape[0], x_j.shape[0]))
        for i in range(x.shape[0]):
            # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
            for j in range(x_j.shape[0]):
                K[i,j] = self.variance * np.exp(-self.gamma.dot((x[i,:]-x_j[j,:])**2))
        return K


    def initialize_GP(self, X, y):
        ## X is a matrix (T,d) of initial T measurements we have results for

        # self.mu = 0
        # self.sigma = self.kernel(x, x_j)

        self.X_t = X
        self.y = y

        T = self.X_t.shape[0]
        a = self.x_star.shape[0]

        self.test_count = np.zeros(a)

        self.K_t = self.kernel(self.X_t, self.X_t)
        self.k_star = self.kernel(self.X_t, self.x_star)

        self.A = np.linalg.inv(self.K_t + self.eta**2 * np.eye(T))
        self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(a) - self.k_star.T @ self.A @ self.k_star
        ### TODO: rewrite sigma computation to be every a not matrix mult
        self.sigma = np.diagonal(self.sigma)

        self.t = T

    def update_obs(self, x, y):
        self.y_t1 = np.array([y])
        self.x_t1 = x[None,...]
       
    def update_GP(self, x, y):
        self.update_obs(x, y)

        ## Can't do internally due to out of memory / invalid array errors from numpy
        self.k_t, self.u, self.phi, f_upd, sigma_upd = update_GP_ext(self.X_t, self.x_t1, self.A, self.x_star, self.eta, self.y, self.y_t1, self.k_star, self.variance, self.gamma)

        # print('Mean f upd: ', np.mean(f_upd))
        # print('Mean sigma upd: ', np.mean(np.diagonal(sigma_upd)))

        self.f = self.f + f_upd
        # self.sigma = self.sigma + np.diagonal(sigma_upd)
        # self.f = self.k_star.T @ self.A @ self.y
        sigma = self.variance * np.eye(self.x_star.shape[0]) - self.k_star.T @ self.A @ self.k_star
        self.sigma = np.diagonal(sigma)

        self.iterate_vars()

    def iterate_vars(self):
        self.y = np.append(self.y, self.y_t1)
        self.X_t = np.append(self.X_t, self.x_t1, axis=0)
        self.k_star = np.append(self.k_star, self.kernel(self.x_t1, self.x_star), axis=0)

        ## update for A
        self.A = self.A + self.phi * np.outer(self.u, self.u)
        self.A = np.vstack((self.A, -self.phi*self.u.T))
        right = np.append(-self.phi*self.u, self.phi)
        self.A = np.column_stack((self.A, right))

        self.t += 1

    def max_acq(self):
        test_pt = np.argmax(self.ucb())
        
        if self.test_count[test_pt] > 5:
            test_pt = np.random.choice(np.arange(self.x_star.shape[0]))
        self.test_count[test_pt] += 1

        return test_pt, self.x_star[test_pt]

    def ucb(self):
        tau = self.d * np.log(self.t + 1e-16)
        # import pdb; pdb.set_trace()
        fcn = self.f + np.sqrt(self.nu * tau) * np.sqrt(self.sigma)
        return fcn

    def stopping(self):
        return np.max(norm.cdf((self.f - np.max(self.f) - 1e-3) / (np.diagonal(self.sigma) + 1e-16)))



def kernel(x, x_j, variance, gamma):
    ## x shape: (T, d) (# tests, # dimensions)
    K = np.zeros((x.shape[0], x_j.shape[0]))
    for i in range(x.shape[0]):
        # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
        for j in range(x_j.shape[0]):
            K[i,j] = variance * np.exp(-gamma.dot((x[i,:]-x_j[j,:])**2))
    return K

def update_GP_ext(X_t, x_t1, A, x_star, eta, y, y_t1, k_star, variance, gamma):

    k_t = kernel(X_t, x_t1, variance, gamma)
    u = A @ k_t
    k_t1 = kernel(x_t1, x_t1, variance, gamma)
    k_star_t1 = kernel(x_t1, x_star, variance, gamma)
    phi = np.linalg.inv(k_t1 + eta**2 - k_t.T.dot(u))
    kuk = k_star.T @ u - k_star_t1.T
    f = np.squeeze(phi * kuk * (y.dot(u) - y_t1))
    sigma = phi * (kuk**2)
    # import pdb; pdb.set_trace()

    return k_t, u, phi, f, sigma 


class SimNeurons():
    ## class to simulate neural tuning curves + responses to visual stimuli
    def __init__(self, N, d, tol = 1):
        self.N = N
        self.d = d
        self.y = []
        self.resp_x = []
        self.resp_z = []
        self.peaks = None
        self.tol = tol

    def set_tuning_x(self, x):
        self.x = x
        self.scale = np.array([l[-2] - l[1] for l in self.x])
        self.min = np.array([l[1] for l in self.x])
        self.max = np.array([l[-2] for l in self.x])

    def gen_tuning_curves(self, type='ind'):
        # currently assuming all single peaked tuning curves
        if type == 'ind':
            means = np.zeros((self.N,self.d))
            covs = np.zeros((self.N,self.d,self.d))
            for i in range(self.d):
                means[:,i] = self.min[i] + np.random.random(size=self.N) * self.scale[i]
                covs[:,i,i] = 5e-2 * self.scale[i]**2 #np.random.random(size=self.N) * self.scale[i]
            for n in range(self.N):
                self.y.append(multivariate_normal(mean=means[n], cov=covs[n]))
            self.peaks = means
        elif type == 'corr':
            pass

    def record_response(self, x, z):
        self.resp_x.append(x)
        self.resp_z.append(z)

    def sample(self, x_sample):
        z = np.zeros(self.N)
        for i,y in enumerate(self.y):
            z[i] = 20 * y.pdf(x_sample) / y.pdf(self.peaks[i])
            # z[i] = np.random.poisson(fr)
        self.record_response(x_sample, z)
        return z

    def verify_sln(self, peaks, n):
        # peaks shape: (N, d)
        dists = np.abs(peaks - self.peaks[n])#/self.scale
        # import pdb; pdb.set_trace()
        count = np.count_nonzero(dists < self.tol)
        return dists, count



if __name__ == "__main__" :

    np.random.seed(42)

    ##################

    ## Neurons
    N = 20      # number of neurons
    d = 3       # number of dimensions in tuning curve

    #### dimensions: direction, speed, spatial frequency, contrast
    x1 = np.arange(15) #np.linspace(0, 350, 15).astype(np.int)
    x2 = np.arange(10) #np.around(np.linspace(0.02, 0.12, num=10), decimals=3)
    x3 = np.arange(8) #np.linspace(10,80,num=8).astype(np.int)
    x4 = np.arange(6)

    SimPop = SimNeurons(N, d, tol = np.array([l[1]-l[0] for l in [x1,x2,x3]]))

    SimPop.set_tuning_x([x1,x2,x3]) #,x3,x4])

    SimPop.gen_tuning_curves()

    ### visualize the generated tuning curves in first 2 dim
    xs = np.meshgrid(*[x1,x2,x3]) #,x3,x4])
    x_star = np.empty(xs[0].shape + (d,))
    for i in range(d):
        x_star[...,i] = xs[i]

    x_star = x_star.reshape(-1, d)      #shape (a,d) where a is all possible test points
    print('Number of possible test points to optimize over: ', x_star.shape[0])

    # import pdb; pdb.set_trace()

    # z = SimPop.sample(pos)

    # plt.contourf(pos[:,:,0,0,0], pos[:,:,0,0,1], np.squeeze(z[8][:,:,0,0]))
    # plt.show()

    ####################
    # Begin by observing some data
    # for _ in range(200):
    #     s1 = np.random.choice(x1)
    #     s2 = np.random.choice(x2)
    #     s3 = np.random.choice(x3)
    #     s4 = np.random.choice(x4)
    #     SimPop.sample(np.array([s1])) #,s2,s3,s4]))

    # plt.plot(np.array(SimPop.resp_x)[:,0], np.array(SimPop.resp_z)[:,0], '.')
    # plt.show()
    # import pdb; pdb.set_trace()

    #################### 

    ## For each neuron we optimize, use a new optimizer
    # params
    gamma = 1 / SimPop.max #2e-1 * 1/SimPop.max       #np.array([5e-4, 1e3]) 
    var = 1e0
    nu = 1e-1
    eta = 1e-2

    # n_optim = 0         # index of neuron currently being optimized
    # print('Peak of this neuron: ', SimPop.peaks[0])

    optim = Optimizer(gamma, var, nu, eta, x_star)

    ## initial test points
    init_T = 4
    X0 = np.zeros((init_T,d))
    X0[:,0] = np.array([0, 10, 2, 5]) #np.array([0, 175, 100, 275]) #, 50, 225, 125, 325])
    X0[:,1] = np.array([2, 4, 8, 1]) #np.array([0.02, 0.05 , 0.1, 0.05])
    X0[:,2] = np.array([0, 5, 3, 7]) #np.array([20, 40, 60, 80])
    # X0[:,3] = np.array([1, 3, 5, 2])

    for n_optim in range(N):
        
        if optim:
            del optim

        for i in range(init_T):
            SimPop.sample(X0[i])
        
        print('Peak of this neuron: ', SimPop.peaks[n_optim])
        y0 = np.array(SimPop.resp_z)[:X0.shape[0],n_optim]

        optim = Optimizer(gamma, var, nu, eta, x_star)
        optim.initialize_GP(X0, y0)
        # print('Current max f :', np.max(optim.f))

        ## For checking kernel:
        # if n_optim == 0:
        #     ks = optim.kernel(optim.x_star, np.array([7, 5])[None,...])
        #     kr = np.reshape(np.squeeze(ks), (15,10), order='F')
        #     plt.imshow(kr)
        #     plt.show()

        for cnt in range(40):
            ind, xt_1 = optim.max_acq()
            # print('Next point ', xt_1, ' with expected value ', optim.ucb()[ind])
            # plt.figure()
            # plt.plot(optim.f, 'k')
            # plt.plot(optim.f+optim.sigma, 'k--')
            # plt.plot(optim.f-optim.sigma, 'k--')
            # plt.plot(optim.ucb(), 'b')
            # plt.show()

            y = SimPop.sample(xt_1)
            optim.update_GP(xt_1, y[n_optim])
            # print('New max f :', np.max(optim.f), ' at ', x_star[np.argmax(optim.f)])

            dists, count = SimPop.verify_sln(optim.x_star[np.argmax(optim.f)], n_optim)
            if count > (d-1):
                print('------ dists: ', dists)
                print('------------------- used ', cnt, ' tests')
                break

        # print('Final value of dists: ', dists)
        # plt.figure()
        # fr = np.reshape(optim.f, (x1.shape[0], x2.shape[0]), order='F')
        # plt.imshow(fr)
        # plt.figure()
        # sr = np.reshape(optim.sigma, (x1.shape[0], x2.shape[0]), order='F')
        # plt.imshow(sr)
        # plt.figure()
        # ur = np.reshape(optim.ucb(), (x1.shape[0], x2.shape[0]), order='F')
        # plt.imshow(ur)
        # plt.show()

        # import pdb; pdb.set_trace()
        # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # ax[0].plot(x1, optim.f[:,0], 'k.-')
        # ax[0].plot(np.array(SimPop.resp_x)[:,0], np.array(SimPop.resp_z)[:,n_optim], 'b.', markersize=10)
        # ax[1].plot(x2, optim.f[:,1], 'k.-')
        # ax[1].plot(np.array(SimPop.resp_x)[:,1], np.array(SimPop.resp_z)[:,n_optim], 'b.', markersize=10)
        # plt.show()

    # ####
    # ## next neuron
    # del optim

    # n_optim = 1
    # print('Peak of this neuron: ', SimPop.peaks[n_optim])
    # y0 = np.array(SimPop.resp_z)[:X0.shape[0],n_optim]

    # optim = Optimizer(gamma, var, nu, eta, x_star)
    # optim.initialize_GP(X0, y0)
    # print('Current max f :', np.max(optim.f))

    # for _ in range(10):
    #     ind, xt_1 = optim.max_acq()
    #     print('Next best point ', xt_1, ' with expected value ', optim.ucb()[ind])

    #     y = SimPop.sample(xt_1)
    #     optim.update_GP(xt_1, y[n_optim])
    #     print('New max f :', np.max(optim.f), ' at ', x_star[np.argmax(optim.f)])

    # # plt.plot(optim.ucb())
    # # plt.show()

    import pdb; pdb.set_trace()

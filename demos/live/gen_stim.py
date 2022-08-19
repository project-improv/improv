import numpy as np

x1 = np.linspace(0,360,endpoint=False,num=24) #np.array([0, 45, 90, 135, 180, 225, 270, 315]) #np.linspace(0,350,num=15).astype(np.int)
x2 = np.linspace(0,360,endpoint=False,num=24)
# x2 = np.array([0.0001, 0.0002, 0.0005, 0.0010, 0.002, 0.005, 0.010, 0.015, 0.020, 0.025 , 0.03  , 0.035 , 0.04  , 0.045 , 0.05  ])
#np.around(np.linspace(0.0001, 0.05, num=20), decimals=4) #np.array([0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.1125]) #np.around(np.linspace(0.02, 0.1, num=4), decimals=2)
# x3 = np.array([5, 15, 30, 45, 60]) #np.linspace(15,45,num=11).astype(np.int) #np.array([5, 15, 30, 45, 60]) #np.linspace(10,80,num=8).astype(np.int)
# x4 = np.arange(5)

labels = ['direction', 'spatial_freq', 'speed', 'contrast']

np.save('stimuli.npy', [x1,x2]) #,x3,x4])
np.save('labels.npy', labels)
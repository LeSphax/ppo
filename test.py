import gym
import gym_ui
import time
import random
import numpy as np

array = [[[1,1],[2,2]], [[3,3],[4,4]], [[5,5],[6,6]]]
arr = np.asarray(array)
print(arr)
print(np.shape(arr))

reshaped = np.reshape(arr, [-1, 2])
print(reshaped)
print(np.shape(reshaped))

back = np.reshape(reshaped, [3,2,2])
print(back)
print(np.shape(back))
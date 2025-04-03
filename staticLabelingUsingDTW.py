import scipy
from scipy import ndimage
from PIL import Image, ImageFilter
import random
import matplotlib.pyplot as plt
from csv import writer
from dtwParallel import dtw_functions
from scipy.spatial import distance as d
import numpy as np
import time
import numpy as numpy
import pandas as pd
import sys
import struct
import numpy as np

def batc(x,seq,index):
    # q = []
    q = x[index:index + 500]#-int((random.random()*2-1)*400)]   ### variation of the sample size ####<<<<<<<<<<<<<<<CHANGED FROM 300 TO 500

    return q

with open(("/home/gagandeep/project/goodSlant.BIN"), "rb") as f:
    binary_data = f.read()

integers = struct.unpack('h' * (len(binary_data) // 2), binary_data)
l = integers

del integers

l = np.reshape(l,(-1,6))
ref = np.zeros((300,2))

ref[:,0] = l[37000:37500,1]#<<<<<<<<<<<<<<<CHANGED FROM 300 TO 500
ref[:,1] = l[37000:37500,2]#<<<<<<<<<<<<<<<CHANGED FROM 300 TO 500
# print(ref)
plt.figure(figsize=(20,6))
plt.plot(ref[:,:])
plt.show()
ref = ref[:,:]###############################


for i in range(np.shape(ref)[1]):
#     img = np.array(img)
    val = ref[:,i]
    value = scipy.ndimage.median_filter(val, size = 201)
    value = scipy.ndimage.gaussian_filter(value, sigma = 300)
    ref[:,i] = value

plt.figure(figsize=(20,6))
plt.plot(ref[:,:])
plt.show()
print("median filtering of ref done")

# Open the binary file in read-binary mode
with open(("/home/gagandeep/project/80_45slant.BIN"), "rb") as f:
    # Read the binary data as a byte string
    binary_data = f.read()

integers = struct.unpack('h' * (len(binary_data) // 2), binary_data)


l = integers
del integers

#print(np.shape(l))
l = np.reshape(l,(-1,6))
print(np.shape(l))
print("****")

################################################################################
data = np.zeros((len(l),2))

data[:,0] = l[:,1]
data[:,1] = l[:,2]
del l
data = data[:,:]###########################

# ref = data
for i in range(np.shape(data)[1]):
#     img = np.array(img)
    val = data[:,i]
    value = scipy.ndimage.median_filter(val, size = 201)
    value = scipy.ndimage.gaussian_filter(value, sigma = 300)
    data[:,i] = value


print("median filtering of data done")


info = []
array = []
seq = ref##################### DONE ABOVE ##############
#data = data[::5,:]######### DONE ABOVE ############
print(np.shape(data))
print(np.shape(seq))


count = 0
print("starting...")

def task_function(data, seq, index):

    print(f'started {index//3000}')
    for i in range(index, index + 13500, 300):## 25000: range for each function to compute; 400: step distance b/w samples
        y = batc(data,seq,i)

        val = int(dtw_functions.dtw(seq, y, type_dtw="d", local_dissimilarity=d.euclidean, MTS=True, n_threads = 32))

        sVal, eVal = i,len(y)+i
        with open('slantDetection300.csv', 'a') as f_object:

            writer_object = writer(f_object)
            writer_object.writerow([val,sVal,eVal])
            f_object.close()
        print(f"done {index//3000}")
        
        


import numpy as np
splits = np.arange(60)*13500   ### make it 55 insted of 40
print(splits)



import concurrent.futures
import multiprocessing

# Define the function that will perform the task


if __name__ == "__main__":
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
#         splits = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000]
        
        results = [executor.submit(task_function,data,seq,i) for i in splits]
        



num_cores = multiprocessing.cpu_count()
print(num_cores)


import multiprocessing
import time
q = time.time()
import numpy as np
import time
import RPi.GPIO as GPIO
from time import sleep
import tflite_runtime.interpreter as tflite#import tensorflow as tf
from serial import Serial
import time
import numpy as np
import struct
import threading
import serial.tools.list_ports
from num2words import num2words
from subprocess import call

#############################################################

# Define the filenames
input_filename = "/webpage.txt"
output_filename = "index.html"

# Step 1: Read the text file
with open(input_filename, 'r') as file:
    text_content = file.read()

# Step 2: Convert to HTML
html_content = f'{text_content}'

# Step 3: Write to an HTML file
with open(output_filename, 'w') as file:
    file.write(html_content)
    
##############################################################

l = threading.Lock()

print(f"Took {time.time()-q} seconds to start inference")

def code1():

    print("starting")

    ports = serial.tools.list_ports.comports ()
    serialInst = serial.Serial() 
    portList = []


    for onePort in ports:
        portList.append(str (onePort))
        #print(str (onePort))
        if str (onePort[1]) == "Nicla Vision":
            path = str (onePort[0])
            print(f'path is {path}')


    ser = Serial(port = path, baudrate = 152000)

    
    # delay = 0.25
    smooothingFactor = 25
    smoothingFactorOfDynamic = 1  ##keep this small for good peak detection
    timeForObservation = 3## time for fall observation in seconds
    gRatio = 4         ## for scaling down the value of "g" for fall detection
    tolerance = 0.9     ## between 0 and 1 for fall detection
    fallState = 0
    sizeOfFallDetection = timeForObservation * 1000 // 45

    fallMatrix = np.zeros((sizeOfFallDetection))
    
    def updateFall(fallState, fallMatrix = fallMatrix):
        fallMatrix[1:] = fallMatrix[:-1]
        fallMatrix[0] = fallState
        return fallMatrix

    def push(newVal, oldVal):
        oldVal[1:] = oldVal[:-1]
        oldVal[0] = newVal
        return oldVal

    result = np.zeros((smooothingFactor,6))
    x = [0,0,0,0,0,0]

    while 1:
        b = time.time()
        if ser.inWaiting():
            a = time.time()

            binary = ser.read(12*300)
            integers1 = struct.unpack('h' * (len(binary) // 2), binary)
            x = np.reshape(integers1, (300, 6))
    
            input_data = np.array(x)
            val = np.sum(input_data, axis=0)[0]
            if val < 600000/gRatio:
                fallState = 1
            else:
                fallState = 0
            fallMatrix = updateFall(fallState)

            if np.sum(fallMatrix) < tolerance * sizeOfFallDetection:
                fallenState = 1

                x = np.reshape(x, (1,300,6))##############################<<<<<<<<<
                input_data = np.array(x, dtype=np.float32)
        
                interpreter = tflite.Interpreter(model_path="model.tflite")#interpreter =tf.lite.Interpreter(model_path="/home/gagandeep/finalModel2024.tflite")
                interpreter.allocate_tensors()
        
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details() 
        
        
                interpreter.resize_tensor_input(input_details[0]['index'], ((len(input_data)), 300,6))###############<<<<<<<
                interpreter.resize_tensor_input(output_details[0]['index'], (len(input_data), 1))
                interpreter.allocate_tensors()
        
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
        
                input_data = input_data.astype(np.float32) # This was missing## for all odels except MoE use ###np.float32###int64
        
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
        
                y = interpreter.get_tensor(output_details[0]['index']).round(3)[0]

                x = list(y[:])
                temp = x
                result = push(x, result)
                x = list(np.sum(result, axis = 0)/smooothingFactor)
                x[4] = sum(result[:smoothingFactorOfDynamic,4])/smoothingFactorOfDynamic  #### smoothing of dynamic implemented independently

                with l:
                    with open('/var/www/html/float_value.txt', 'w') as f:
                        f.write(str(x))
                    
                with open('/var/www/html/report.txt', 'a') as f:
                    f.write(str(x))
            else: 
                with l:
                    with open('/var/www/html/float_value.txt', 'w') as f:
                        f.write("[1,1,1,1,0,0]")


def code2():

    def push(newVal, oldVal):
        if newVal != oldVal[0]:
            oldVal[1:] = oldVal[:-1]
            oldVal[0] = newVal
            return oldVal
        return oldVal
        
        
    def sign(val):
        if val // (abs(val)+1) > 0:
            return 1
        return -1
        
    def polarityCheck(arr):
        if sign(arr[len(arr)//2]) == sign(arr[-2]) and sign(arr[len(arr)//2]) == sign(arr[1]):
            if sign(arr[len(arr)//2]) != sign(arr[-1]):
                if sign(arr[len(arr)//2]) != sign(arr[0]):
                    return 0, 0
                else:
                    return 1, 0
            else:
                if sign(arr[len(arr)//2]) != sign(arr[0]):
                    return 0, 1
                else:
                    return 1, 1
            
    
    def peakDet(runningPeaks):### Recommed size of runningPeaks = 5
        if abs(runningPeaks[len(runningPeaks)//2]) > abs(runningPeaks[1]) and abs(runningPeaks[len(runningPeaks)//2]) > abs(runningPeaks[-2]):
            polarity1, polarity2 = polarityCheck(runningPeaks)
            if polarity1 == 0:
                runningPeaks[0] = 0
            if polarity2 == 0:
                runningPeaks[-1] = 0
            
            if abs(runningPeaks[len(runningPeaks)//2]) >= max(abs(runningPeaks[0]), abs(runningPeaks[-1])):
                return 1, runningPeaks[len(runningPeaks)//2]
        return 0, None
        
    
    pin = 12
    pin2 = 13
    GPIO.setwarnings(False)			#disable warnings
    GPIO.setmode(GPIO.BCM)		#set pin numbering system
    GPIO.setup(pin,GPIO.OUT)
    GPIO.setup(pin2,GPIO.OUT)
    import numpy as np
    #print("waiting 1 seconds for module to load")
    time.sleep(1)
    print("starting GPIO operation")
    count = 0
    high = 0
    med = 0
    low = 0
    tm = 10 ### time window for observation of the dynamic motion
    thresholdValueLow = 0.20
    thresholdValueMed = 0.30
    thresholdValueHigh = 0.40
    sleepTime = 0.25
    timePeriod = 0.150
    runningPeaks = [0, 0, 0, 0, 0]## here the size of runningPeaks is defined
    while 1:
        values = []
        with l:
            with open('/var/www/html/float_value.txt', 'r') as f:
                content = f.readlines()
                f.close()

        if content != []:
          content = content[0][1:-1]

        val = content.split(",")
        for i in val:
            values.append(float(i))
        
        storedPeaks = runningPeaks[:]
        updatedPeaks = push(values[4], runningPeaks)
        updateFlag = 0
        if storedPeaks != updatedPeaks:
            runningPeaks = updatedPeaks
            updateFlag = 1
            
        peakStatus, peakValue = peakDet(runningPeaks)

        if values[5]>0.75:
#            print("inside of if")
            pin = 12
            val = abs(values[4])
            if count == 0:
                t = time.time()
            if time.time() - t > tm:
                cmd_beg= 'espeak '
                cmd_end= ' | aplay /home/gagandeep/Text.wav  2>/dev/null' # To play back the stored .wav file and to dump the std errors to /dev/null
                cmd_out= '--stdout > /home/gagandeep/Text.wav ' # To store the voice file
                if high>med and high>low:
                    word = "excessively"
                elif med>high and med>low:
                    word = "perfectly"
                else:
                    word = "less"
                maxCount = max(high, med, low)
                maxCount = int(maxCount/(high + med + low + 1)*100)
                text = f"You have moved the cane {word}, {maxCount} percent in the last {tm} seconds"
                print(text)

                #Replacing ' ' with '_' to identify words in the text entered
                text = text.replace(' ', '_')

                #Calls the Espeak TTS Engine to read aloud a Text 
                call([cmd_beg+cmd_out+text+cmd_end], shell=True)
                t = time.time()
                count = 0
                high, med, low = 0, 0, 0
            
            if peakStatus==1 and updateFlag==1:
                count+=1
                updateFlag = 0
                print(count)
                if abs(peakValue) > thresholdValueMed:
                    high += 1
                elif abs(peakValue) <= thresholdValueMed and abs(peakValue) >= thresholdValueLow:
                    med += 1
                else:
                    low += 1

        else:
            high, med, low = 0, 0, 0
            slantVal = values[0] + values[3]
            slantVal = 1 if slantVal>1 else slantVal
            slantVal = 1 if values[3]>0.1 else slantVal
            orientationValue = np.sqrt(values[1]**2 + values[2]**2)
            
            if orientationValue > 1:
                orientationValue = 1        
            
            if slantVal >= orientationValue and (values[1]<0.1 and values[2]<0.1):
                val = abs(0.4-slantVal)
                pin = 12
            else:
                val = orientationValue
                pin = 13
        
            case = 0
            if val>0.40:
                case = 3
            elif val>0.25 and val<0.40:
                case = 2
            elif val>0.10:
                case = 1
            else:
                case = 0

            GPIO.output(pin,GPIO.HIGH)
            time.sleep(timePeriod*case/3)
            GPIO.output(pin,GPIO.LOW)
            time.sleep(timePeriod - timePeriod*case/3)
        #case:1)>80 2)>50 3)>15 4)>0

if __name__ == "__main__":
#     # Create two processes, each running a different code

    
    thread_one = threading.Thread(target=code1)
    thread_two = threading.Thread(target=code2)
    # Start both threads

    
    thread_one.start()
    thread_two.start()


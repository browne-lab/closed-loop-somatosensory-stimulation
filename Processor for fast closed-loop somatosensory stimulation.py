"""
Title: Processor for fast closed-loop somatosensory stimulation

Description:
This script converts keypoints to mirror galvanometer voltages, targets and triggers a laser

Attribution:
This file is based on work originally found in the DeepLabCut Toolbox (deeplabcut.org, https://github.com/DeepLabCut/DeepLabCut-live/blob/master/example_processors/MouseLickLED/lick_led.py) by A. & M. Mathis Labs. It was modified by Liam E. Browne (https://github.com/lebrowne) and Isobel Parkes (https://github.com/Isoparkes) as an example for Parkes et al 2024.

Date:
Created on: July 1, 2024
Last updated: July 2, 2024

Original License:
This work includes modifications to code licensed under the GNU Lesser General Public License v3.0.
For more information, see <https://www.gnu.org/licenses/lgpl-3.0.html>.

Additional Contributions License:
The modifications made by Liam E. Browne and Isobel Parkes are licensed under the MIT License. For more details, see <https://opensource.org/licenses/MIT>.
"""

import numpy as np
import pickle
import time
import math
import nidaqmx
import serial
import struct
from nidaqmx.constants import LineGrouping
from dlclive.processor.processor import Processor

dict_path = r'path\dict.pickle' ### Add path for pixel-to-voltage dictionary. This can be generated using Generate pixel-to-voltage map.ipynb 
with open(dict_path, 'rb') as f:
    transformation_matrix = pickle.load(f)  

Task_x = nidaqmx.Task() 
Task_x.ao_channels.add_ao_voltage_chan("Dev2/ao0")
Task_y = nidaqmx.Task()
Task_y.ao_channels.add_ao_voltage_chan("Dev2/ao1")
Task_laser = nidaqmx.Task()
Task_laser.do_channels.add_do_chan("Dev2/port1/line0", line_grouping=LineGrouping.CHAN_PER_LINE)

class FastStim(Processor):
      
    def __init__(self, lik_thresh=0.8):
        super().__init__()
        self.lik_thresh = lik_thresh
        self.trig_times = []
        self.Task_x = Task_x
        self.Task_y = Task_y
        self.Task_laser = Task_laser

    def track_stim(self, frame_time):
        ctime = time.time()
        self.trig_times.append((frame_time, ctime))
        #print(frame_time, ctime)

    def process(self, pose, **kwargs):
              
        if pose[0, 2] > self.lik_thresh: ### Select keypoint here and the below two lines
            pixel_x = pose[0, 0].astype(int)
            pixel_y = pose[0, 1].astype(int)
            # print('from single pulse:', pose[0])

            # Coerce
            if pixel_x > 1919: pixel_x = 1919  ### Modify according to stimuluation area
            if pixel_x < 0: pixel_x = 0
            if pixel_y > 1199: pixel_y = 1199 ### Modify according to stimulation area
            if pixel_y < 0: pixel_y = 0 
        
            xy = transformation_matrix[pixel_x, pixel_y] ### This transforms the keypoint to voltages
        
            self.Task_x.write(xy[0], auto_start=True)
            self.Task_y.write(xy[1], auto_start=True)
            self.Task_laser.write([True], auto_start=True) ### Trigger a laser
            self.track_stim(kwargs["frame_time"])
            time.sleep(0.001)
            self.Task_laser.write([False])
            
        return pose

    def save(self, filename):

        if filename[-4:] != ".npy":
            filename += ".npy"
        arr = np.array(self.trig_times, dtype=float)
        try:
            np.save(filename, arr)
            save_code = True
        except Exception:
            save_code = False

        return save_code

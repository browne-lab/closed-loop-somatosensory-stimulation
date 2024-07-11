"""
Title: Processor for somatosensory stimulation in a maze

Description:
This script converts keypoints to mirror galvanometer voltages, sends a trigger to modify laser frequency, and targets a laser in selected ROIs

Attribution:
This file is based on work originally found in the DeepLabCut Toolbox (deeplabcut.org, https://github.com/DeepLabCut/DeepLabCut-live/blob/master/example_processors/MouseLickLED/lick_led.py) by A. & M. Mathis Labs. It was modified by Liam E. Browne (https://github.com/lebrowne) and Isobel Parkes (https://github.com/Isoparkes) as an example for Parkes et al 2024.

Date:
Created on: July 6, 2024
Last updated: July 11, 2024

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

from dlclive.processor.processor import Processor

dict_path = r'path\dict.pickle' ### Add path for pixel-to-voltage dictionary. This can be generated using Generate pixel-to-voltage map.ipynb 
with open(dict_path, 'rb') as f:
    transformation_matrix = pickle.load(f)

corridor_1_stim = np.load(r'path\ROI_1.npy') ### Add path to corridor ROI where stimulations will be targeted (ROI should be begin after corridor boundary to ensure all keypoints are in the corridor before stimulation)
corridor_2_stim = np.load(r'path\ROI_2.npy')

corridor_high_freq = np.load(r'path\ROI_1_extended.npy') ### Add path to corridor ROI where high frequency stimulation will be targeted
                                                        ### ROI is extended to the corridor boundary to ensure the laser is in the correct frequency state before stimulation is targeted to selected keypoint

Task_x = nidaqmx.Task()
Task_x.ao_channels.add_ao_voltage_chan("Dev2/ao0")
Task_y = nidaqmx.Task()
Task_y.ao_channels.add_ao_voltage_chan("Dev2/ao1")

Task_laser = nidaqmx.Task()
Task_laser.do_channels.add_do_chan("Dev2/port1/line1")

Task_reward1 = nidaqmx.Task()
Task_reward1.di_channels.add_di_chan("Dev2/port0/line3")
Task_reward2 = nidaqmx.Task()
Task_reward2.di_channels.add_di_chan("Dev2/port0/line1")

Task_reset_reward1 = nidaqmx.Task()
Task_reset_reward1.do_channels.add_do_chan("Dev2/port1/line0")
Task_reset_reward2 = nidaqmx.Task()
Task_reset_reward2.do_channels.add_do_chan("Dev2/port1/line2")

class LaserTrack(Processor):

    def __init__(self, lik_thresh=0.8):

        super().__init__()
        self.lik_thresh = lik_thresh
        self.Task_x = Task_x
        self.Task_y = Task_y
        self.stim_prob = 0
        self.r1 = 0 ### Reward 1 = r1
        self.r2 = 0 ### Reward 2 = r2
        self.reward1_t = 0
        self.reward2_t = 0
        self.reset = 0

    def process(self, pose, **kwargs):

        t = time.time_ns() / 1000000000

        pixel_x = pose[2, 0].astype(int) ### Select keypoint here and the below line
        pixel_y = pose[2, 1].astype(int)

        # Coerce pixels
        if (pixel_x > 1919): pixel_x = 1919 ### Modify according to stimuluation area
        if (pixel_x < 0): pixel_x = 0
        if (pixel_y > 1199): pixel_y = 1199 ### Modify according to stimulation area
        if (pixel_y < 0): pixel_y = 0 

        xy = transformation_matrix[pixel_x, pixel_y] ### This transforms the keypoint to voltages
        
        x = xy[0]
        y = xy[1]
        
        # Coerce voltages
        if (x > 10): x = 10 ### Modify according to stimuluation area
        if (x < -10): x = -10
        if (y > 10): y = 10 ### Modify according to stimuluation area
        if (y < -10): y = -10 

        if not corridor_1_stim[pixel_y, pixel_x] | corridor_2_stim[pixel_y, pixel_x]: 
            x = 1 ### Modify pixel values to offset the laser here and the below line
            y = -1

        if corridor_high_freq[pixel_y, pixel_x]:
            Task_laser.write(True) ### Sends a trigger to switch the laser from default low frequency state to high frequency state
        if corridor_1_stim[pixel_y, pixel_x]: 
            x = x ### Moves mirror galvanometers from the offset position to seleted keypoint
            y = y
                
        if corridor_2_stim[pixel_y, pixel_x]: 
            x = x ### Moves mirror galvanometers from the offset position to seleted keypoint
            y = y

        if pose[0,2] > self.lik_thresh:
            self.Task_x.write(x, auto_start=True)
            self.Task_y.write(y, auto_start=True)
        
        if self.r1 == 0: 
            if Task_reward1.read() == True: 
                self.reward1_t = t ### Records a timestamp for when a reward is collected
                # print('reward 1 timestamp:', self.reward1_t)
                self.r1 = 1
        if self.r2 == 0:
            if Task_reward2.read() == True:
                self.reward2_t = t ### Records a timestamp for when a reward is collected
                # print('reward 2 timestamp:', self.reward2_t)
                self.r2 = 1
            
        if (pixel_y > 900) & (290 < pixel_x < 1120): ### Modify according to dimensions of maze
            Task_reset_reward1.write(True) ### Send a trigger to reset reward ports here and below line
            Task_reset_reward2.write(True)
            Task_laser.write(False)
            self.r1 = 0
            self.r2 = 0
  
        if pixel_y < 900: ### Modify according to dimensions of maze
            Task_reset_reward1.write(False)
            Task_reset_reward2.write(False)
            self.reset = 0

        return pose

    def save(self, file=None):

        if file:
            try:
                pickle.dump({"reward1_timestamp": self.reward1_t, "reward2_timestamp": self.reward2_t}, open(file, "wb"))
                save_code = True
            except Exception:
                save_code = False
        return save_code
"""
Title: Processor for random-access multi-animal somatosensory stimulation

Description:
This script generates a random trial sequence for each animal, sends a signal to modulate laser intensity dependent on the trial sequence, and converts keypoints to mirror galvanometer voltages to target and trigger a laser.
This processor ultilises our modified DLC-Live! scripts: Multi-chamber real-time pose estimation.ipynb 

Attribution:
This file is based on work originally found in the DeepLabCut Toolbox (deeplabcut.org, https://github.com/DeepLabCut/DeepLabCut-live/blob/master/example_processors/MouseLickLED/lick_led.py) by A. & M. Mathis Labs. It was modified by Isobel Parkes (https://github.com/Isoparkes) as an example for Parkes et al 2024.

Date:
Created on: July 6, 2024
Last updated: July 11, 2024

Original License:
This work includes modifications to code licensed under the GNU Lesser General Public License v3.0.
For more information, see <https://www.gnu.org/licenses/lgpl-3.0.html>.

Additional Contributions License:
The modifications made by Isobel Parkes are licensed under the MIT License. For more details, see <https://opensource.org/licenses/MIT>.
"""

import os
import numpy as np
import pandas as pd
import pickle
import time
import math
import nidaqmx
import random
import euler
from nidaqmx.stream_writers import CounterWriter
from nidaqmx.constants import *

from dlclive.processor.processor import Processor

dict_path = r'path\dict.pickle' ### Add path for pixel-to-voltage dictionary. This can be generated using Generate pixel-to-voltage map.ipynb 
with open(dict_path, 'rb') as f:
    transformation_matrix = pickle.load(f)

Task_x = nidaqmx.Task()
Task_x.ao_channels.add_ao_voltage_chan("Dev2/ao0")
Task_y = nidaqmx.Task()
Task_y.ao_channels.add_ao_voltage_chan("Dev2/ao1")

Task_laser = nidaqmx.Task()
Task_laser.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr0") 
Task_laser.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
cw_laser = CounterWriter(Task_laser.out_stream, True)

Task_laser_intensity = nidaqmx.Task()
Task_laser_intensity.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr1")
Task_laser_intensity.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
cw_intensity = CounterWriter(Task_laser_intensity.out_stream, True)

freq_analog_modulation = {0:100, 1:200, 2:300, 3:400, 4:500, 5:600} ### Modify according to desired number of laser intensities (5 laser intensities are used in Parkes et al)

class RandomAcessStim(Processor):

    def __init__(self, lik_thresh=0.9):
        
        super().__init__()
        self.lik_thresh = lik_thresh
        self.trial_t_leftpaw = []
        self.trial_t_rightpaw = []
        self.Task_x = Task_x
        self.Task_y = Task_y
        self.Task_laser = Task_laser
        self.cw_laser = cw_laser
        self.Task_laser_intensity = Task_laser_intensity
        self.cw_intensity = cw_intensity
        self.generate_trial_order = 0
        self.trial = False
        self.task_ready = False
        self.t0 = 0

        self.stim_trial_leftpaw = []
        self.trial_idx_leftpaw = [0, 0, 0, 0, 0, 0, 0, 0, 0] ### Modify according to the number of chambers (animals)
        self.stim_trial_rightpaw = []
        self.trial_idx_rightpaw = [0, 0, 0, 0, 0, 0, 0, 0, 0] ### Modify according to the number of chambers (animals)

        self.stim_count_leftpaw = [0, 0, 0, 0, 0, 0, 0, 0, 0] ### Modify according to the number of chambers (animals)
        self.stim_count_rightpaw = [0, 0, 0, 0, 0, 0, 0, 0, 0] ### Modify according to the number of chambers (animals)

        self.t = time.time_ns() / 1000000000
        self.stim_timestamps = np.empty(9); self.stim_timestamps.fill(self.t)
        self.stim_timestamps = list(self.stim_timestamps)

    def process(self, pose, **kwargs):
        
        if self.generate_trial_order == 0:

            for i in np.arange(9): ### Modify according to the number of chambers (animals)
                stim_trial = euler.Euler(stimuli=6, stim_repeat=False).get_sequence(0) ### Modify according to the desired number of stimulation trials
                stim_trial = [int(n) for n in stim_trial]
                self.stim_trial_leftpaw.append(stim_trial)
            save_stim_trial_leftpaw = pd.DataFrame(self.stim_trial_leftpaw)
            save_stim_trial_leftpaw.to_csv(r'path\stim_trial_leftpaw.csv') ### Save the stimulation trial order

            for i in np.arange(9): ### Modify according to the number of chambers (animals)
                stim_trial = euler.Euler(stimuli=6, stim_repeat=False).get_sequence(0)
                stim_trial = [int(n) for n in stim_trial]
                self.stim_trial_rightpaw.append(stim_trial)
            save_stim_trial_rightpaw = pd.DataFrame(self.stim_trial_rightpaw)
            save_stim_trial_rightpaw.to_csv(r'path\stim_trial_rightpaw.csv') ### Save the stimulation trial order
            
            self.generate_trial_order = 1

        t = time.time_ns() / 1000000000

        selected_chamber = pose[0, 3].astype(int)

        pixel_x_leftpaw = pose[0, 0].astype(int) ### Select keypoint here and the below line
        pixel_y_leftpaw = pose[0, 1].astype(int)

        # Coerce pixels
        if (pixel_x_leftpaw > 1919): pixel_x_leftpaw = 1919 ### Modify according to stimuluation area
        if (pixel_x_leftpaw < 0): pixel_x_leftpaw = 0
        if (pixel_y_leftpaw > 1199): pixel_y_leftpaw = 1199 ### Modify according to stimuluation area
        if (pixel_y_leftpaw < 0): pixel_y_leftpaw = 0 
        
        xy_leftpaw = transformation_matrix[pixel_x_leftpaw, pixel_y_leftpaw] ### This transforms the keypoint to voltages

        x_leftpaw = xy_leftpaw[0]
        y_leftpaw = xy_leftpaw[1]
        
        # Coerce voltages
        if (x_leftpaw > 10): x_leftpaw = 10 ### Modify according to stimuluation area
        if (x_leftpaw < -10): x_leftpaw = -10
        if (y_leftpaw > 10): y_leftpaw = 10 ### Modify according to stimuluation area
        if (y_leftpaw < -10): y_leftpaw = -10   

        pixel_x_rightpaw = pose[2, 0].astype(int) ### Select keypoint here and the below line
        pixel_y_rightpaw = pose[2, 1].astype(int)

        # Coerce pixels
        if (pixel_x_rightpaw > 1919): pixel_x_rightpaw = 1919 ### Modify according to stimuluation area
        if (pixel_x_rightpaw < 0): pixel_x_rightpaw = 0
        if (pixel_y_rightpaw > 1199): pixel_y_rightpaw = 1199 ### Modify according to stimuluation area
        if (pixel_y_rightpaw < 0): pixel_y_rightpaw = 0 
        
        xy_rightpaw = transformation_matrix[pixel_x_rightpaw, pixel_y_rightpaw]

        x_rightpaw = xy_rightpaw[0]
        y_rightpaw = xy_rightpaw[1]
        
        # Coerce voltages
        if (x_rightpaw > 10): x_rightpaw = 10 ### Modify according to stimuluation area
        if (x_rightpaw < -10): x_rightpaw = - 10
        if (y_rightpaw > 10): y_rightpaw = 10 ### Modify according to stimuluation area
        if (y_rightpaw < -10): y_rightpaw = - 10 
        
        if selected_chamber >= 0:
            selected_chamber_timestamp = self.stim_timestamps[selected_chamber]
            if (t-selected_chamber_timestamp) > 60: ### Modify according to desired inter-stimulus interval (in seconds)
                if self.stim_count_leftpaw[selected_chamber] <=  self.stim_count_rightpaw[selected_chamber]:
                    if not self.trial:
                        if self.stim_count_leftpaw[selected_chamber] < 18: ### Modify according to desired number of stimulations per paw
                            if pose[0,2] > self.lik_thresh:
                                self.Task_x.write(x_leftpaw, auto_start=True)
                                self.Task_y.write(y_leftpaw, auto_start=True)
                                intensity = self.trial_idx_leftpaw[selected_chamber]
                                stim_trial = self.stim_trial_leftpaw[selected_chamber]
                                intensity = stim_trial[intensity]
                                if self.task_ready == False:
                                    freq = freq_analog_modulation[intensity] 
                                    self.Task_laser_intensity.start()
                                    self.cw_intensity.write_one_sample_pulse_frequency(freq, 0.5, 10) ### Send signal to change the laser intensity
                                    self.t0 = t
                                    time.sleep(0.5) 
                                    self.Task_laser_intensity.close()
                                    self.Task_laser_intensity = nidaqmx.Task()
                                    self.Task_laser_intensity.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr1")
                                    self.Task_laser_intensity.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_intensity = CounterWriter(Task_laser_intensity.out_stream, True)
                                    self.task_ready = True
                                if (intensity == 0) & ((t - self.t0) > 1) & (self.task_ready == True): 
                                    self.Task_laser.start()
                                    self.cw_laser.write_one_sample_pulse_frequency(10, 0.03, 10) ### Triggers a laser pulse
                                    self.trial = True
                                    self.trial_t_leftpaw.append(t)
                                    selected_chamber_timestamp = t
                                    self.stim_timestamps[selected_chamber] = selected_chamber_timestamp
                                    update_stim_count = self.stim_count_leftpaw[selected_chamber]+1
                                    self.stim_count_leftpaw[selected_chamber] = update_stim_count
                                    update_stim_trial_trial_idx = self.trial_idx_leftpaw[selected_chamber]+1
                                    self.trial_idx_leftpaw[selected_chamber] = update_stim_trial_trial_idx
                                    # print('stim #:', self.stim_count_leftpaw)
                                    self.stim_coords_leftpaw = pose
                                    time.sleep(0.05) ### Generates a single laser pulse
                                    self.Task_laser.close()
                                    time.sleep(1)
                                    self.Task_laser = nidaqmx.Task()
                                    self.Task_laser.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr0")
                                    self.Task_laser.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_laser = CounterWriter(Task_laser.out_stream, True)
                                    self.trial = False
                                    self.task_ready = False
                                if (intensity != 0) & ((t - self.t0) > 1) & (self.task_ready == True):
                                    self.Task_laser.start()
                                    self.cw_laser.write_one_sample_pulse_frequency(10, 0.03, 10) ### Triggers a laser pulse train
                                    self.trial = True 
                                    self.trial_t_leftpaw.append(t)
                                    selected_chamber_timestamp = t
                                    self.stim_timestamps[selected_chamber] = selected_chamber_timestamp
                                    update_stim_count = self.stim_count_leftpaw[selected_chamber]+1
                                    self.stim_count_leftpaw[selected_chamber] = update_stim_count
                                    update_stim_trial_trial_idx = self.trial_idx_leftpaw[selected_chamber]+1
                                    self.trial_idx_leftpaw[selected_chamber] = update_stim_trial_trial_idx
                                    # print('stim #:', self.stim_count_leftpaw)
                                    self.stim_coords_leftpaw = pose
                                    time.sleep(10) ### Generates a pulse train (10 seconds)
                                    self.Task_laser.close()
                                    time.sleep(1)
                                    self.Task_laser = nidaqmx.Task()
                                    self.Task_laser.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr0")
                                    self.Task_laser.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_laser = CounterWriter(Task_laser.out_stream, True)
                                    self.trial = False
                                    self.task_ready = False

                else:
                    if not self.trial:
                        if self.stim_count_rightpaw[selected_chamber] < 18: ### Modify according to desired number of stimulations per paw
                            if pose[0,2] > self.lik_thresh:
                                self.Task_x.write(x_rightpaw, auto_start=True)
                                self.Task_y.write(y_rightpaw, auto_start=True)
                                intensity = self.trial_idx_rightpaw[selected_chamber]
                                stim_trial = self.stim_trial_rightpaw[selected_chamber]
                                intensity = stim_trial[intensity]
                                if self.task_ready == False:
                                    freq = freq_analog_modulation[intensity]
                                    self.Task_laser_intensity.start()
                                    self.cw_intensity.write_one_sample_pulse_frequency(freq, 0.5, 10) ### Send signal to change the laser intensity
                                    self.t0 = t
                                    time.sleep(0.5) 
                                    self.Task_laser_intensity.close()
                                    self.Task_laser_intensity = nidaqmx.Task()
                                    self.Task_laser_intensity.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr1")
                                    self.Task_laser_intensity.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_intensity = CounterWriter(Task_laser_intensity.out_stream, True)
                                    self.task_ready = True
                                if (intensity == 0) & ((t - self.t0) > 1) & (self.task_ready == True):
                                    self.Task_laser.start()
                                    self.cw_laser.write_one_sample_pulse_frequency(10, 0.03, 10) ### Triggers a laser pulse
                                    self.trial = True 
                                    self.trial_t_rightpaw.append(t)
                                    selected_chamber_timestamp = t
                                    self.stim_timestamps[selected_chamber] = selected_chamber_timestamp
                                    update_stim_count = self.stim_count_rightpaw[selected_chamber]+1
                                    self.stim_count_rightpaw[selected_chamber] = update_stim_count
                                    update_stim_trial_trial_idx = self.trial_idx_rightpaw[selected_chamber]+1
                                    self.trial_idx_rightpaw[selected_chamber] = update_stim_trial_trial_idx
                                    # print('stim #:', self.stim_count_rightpaw)
                                    self.stim_coords_rightpaw = pose
                                    time.sleep(0.05) ### Generates a single laser pulse
                                    self.Task_laser.close()
                                    time.sleep(1)
                                    self.Task_laser = nidaqmx.Task()
                                    self.Task_laser.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr0")
                                    self.Task_laser.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_laser = CounterWriter(Task_laser.out_stream, True)
                                    self.trial = False 
                                    self.task_ready = False
                                if (intensity != 0) & ((t - self.t0) > 1) & (self.task_ready == True):
                                    self.Task_laser.start()
                                    self.cw_laser.write_one_sample_pulse_frequency(10, 0.03, 10) ### Triggers a laser pulse train
                                    self.trial = True 
                                    self.trial_t_rightpaw.append(t)
                                    selected_chamber_timestamp = t
                                    self.stim_timestamps[selected_chamber] = selected_chamber_timestamp
                                    update_stim_count = self.stim_count_rightpaw[selected_chamber]+1
                                    self.stim_count_rightpaw[selected_chamber] = update_stim_count
                                    update_stim_trial_trial_idx = self.trial_idx_rightpaw[selected_chamber]+1
                                    self.trial_idx_rightpaw[selected_chamber] = update_stim_trial_trial_idx
                                    # print('stim #:', self.stim_count_rightpaw)
                                    self.stim_coords_rightpaw = pose
                                    time.sleep(10) ### Generates a pulse train (10 seconds)
                                    self.Task_laser.close()
                                    time.sleep(1)
                                    self.Task_laser = nidaqmx.Task()
                                    self.Task_laser.co_channels.add_co_pulse_chan_time(counter = "Dev2/ctr0")
                                    self.Task_laser.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                                    self.cw_laser = CounterWriter(Task_laser.out_stream, True)
                                    self.trial = False
                                    self.task_ready = False
                                    
        return pose, self.trial
    
    def save(self, file=None):

        if file:
            try:
                pickle.dump({"trial_timestamp_leftpaw": self.trial_t_leftpaw, "trial_timestamp_rightpaw": self.trial_t_rightpaw}, open(file, "wb"))
                save_code = True
            except Exception:
                save_code = False

        return save_code

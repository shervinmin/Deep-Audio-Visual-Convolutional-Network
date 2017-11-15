#### shervin minaee, summer 2017
#### videos audio extraction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import librosa
import numpy as np
import pickle, time, cv2, glob, h5py
import subprocess as sp
import skvideo.io  ## http://www.scikit-video.org/stable/examples/io.html#ioexamplevideo
import matplotlib.pyplot as plt
import os.path
import moviepy.editor as mp


start_time= time.time()
sys.path.append("/home/shervin/Sherv_Files/New Dataset/video") 


################## Audio mfcc feature extraction, 10 shots per video is used.
################## if there are less than 10 shots, the last one is repeated
file_name= sorted(glob.glob("/home/shervin/Sherv_Files/New Dataset/video/*.mp4"), key=os.path.getsize)


####################################### Loop over videos
previous_shot_mfcc= np.zeros([200,])
Vid_mfcc= list()
num_ad_audioshot= 0   ### 
not_available_info= list()
for video_ind in range( 0, len(file_name)):  ## defected videos
#for video_ind in range( 0, 10):
    print("%d-th video beong processed" %(video_ind))    
    cur_video_name= mp4_files[video_ind]
    cur_video= skvideo.io.vread(cur_video_name)
    cur_audio, cur_sr= librosa.load(cur_video_name)
    cur_num_frame= cur_video.shape[0]
    cur_shot_name= cur_video_name[:-4]+'_shots.txt'
    #if os.path.isfile(cur_shot_name):  ### if the shot-detection info is available
    if 2>1:
        line = list(open(cur_shot_name, 'r'))  
        if len(line)>= 10:   ### if more than 11 shots, get the last 10 shots
            start_shot_ind= len(line)-10
            frame_range= list( range( start_shot_ind, len(line)) )
        else:
            frame_range= list( range( 0, len(line)) )
            for extra_ind in range(10-len(line)):
                frame_range.append( len(line)-1)
        
        for shot_ind in frame_range:    
            cur_line= line[shot_ind]
            cur_line_list= cur_line.split()
            first_frame= int( cur_line_list[0] )
            last_frame = int( cur_line_list[1] )
            mid_frame  = int( cur_line_list[3] )
            
            if (last_frame-first_frame)>=5:
            
                cur_audioshot_first_ind= int( np.floor(first_frame*len(cur_audio)/cur_num_frame ) )
                cur_audioshot_last_ind = int( np.floor(last_frame *len(cur_audio)/cur_num_frame ) )
                cur_audioshot= cur_audio[cur_audioshot_first_ind:cur_audioshot_last_ind]
                new_rate= 5000*cur_sr/len(cur_audioshot)
                cur_audioshot_resampled = librosa.resample(cur_audioshot, cur_sr, new_rate)
                cur_audioshot_mfcc= librosa.feature.mfcc(y=cur_audioshot_resampled, sr= new_rate, n_mfcc=20)
                print("current mfcc dimension=", cur_audioshot_mfcc.shape )
                
                cur_audioshot_mfcc_reshaped= np.reshape( cur_audioshot_mfcc, [cur_audioshot_mfcc.shape[0]*cur_audioshot_mfcc.shape[1],])
                Vid_mfcc.append(cur_audioshot_mfcc_reshaped)
                num_ad_audioshot= num_ad_audioshot+1 
                previous_shot_mfcc= cur_audioshot_mfcc_reshaped
            else:
                print("less than 5 frames in this shot!")
                Vid_mfcc.append(previous_shot_mfcc)
                
    else:
        print("not available video name!")
        not_available_info.append(video_ind)
        

pickle.dump( Vid_mfcc, open( "Vid500_10shots_mfcc.p", "wb" ), protocol= 2 ) 
#pickle.dump( Vid_mfcc, open( "Vid500_10shots_mfcc.p", "wb" ), protocol= pickle.HIGHEST_PROTOCOL )        


tot_time= time.time()
print("program time:", tot_time-start_time)






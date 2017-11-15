#### Shot-extractor from new vide dataset. Movies, serials, News, Sport matches, Documentaries


import numpy as np
import pickle, time, cv2, sys, glob, os
import subprocess as sp
import skvideo.io  ## http://www.scikit-video.org/stable/examples/io.html#ioexamplevideo
import pylab
import imageio
import librosa

sys.path.append("/home/shervin/Sherv_Files/New Dataset/video") 
st_time= time.time()
    

file_name= sorted(glob.glob("/home/shervin/Sherv_Files/New Dataset/video/*.mp4"), key=os.path.getsize)


Vid_keyframes= list()
num_ad_keyframes= 0   ### 2521*5 shots for these 101 ads
num_shots= 0

for video_ind in range(len(file_name)):  ##
    print("%d-th video" %(video_ind))
    cur_video_name= file_name[video_ind]
    cur_shot_name= cur_video_name[:-4]+'_shots.txt'
    
    cur_video= skvideo.io.vread(cur_video_name)
    cur_w= cur_video.shape[1]
    cur_h= cur_video.shape[2]
          
    line = list(open(cur_shot_name, 'r'))
    max_shot= 10
    if len(line)>= max_shot:   ### if more than 11 shots, get the last 10 shots
        start_shot_ind= len(line)-max_shot
        frame_range= list( range( 0, max_shot) )
    else:
        frame_range= list( range( 0, len(line)) )
        for extra_ind in range(max_shot-len(line)):
            frame_range.append( len(line)-1)
    #print("range=", frame_range)
        
    for shot_ind in frame_range:
        #print("%d-th video and %d-th shot" %(video_ind,shot_ind))
        cur_line= line[shot_ind]
        cur_line_list= cur_line.split()
        first_frame= int( cur_line_list[0] )
        last_frame = int( cur_line_list[1] )
        mid_frame  = int( cur_line_list[3] )
        cur_key_frame1= cur_video[ first_frame-1,:,:,:]
        cur_key_frame2= cur_video[ mid_frame-1,:,:,:]
        cur_key_frame3= cur_video[ last_frame-1,:,:,:]
        num_shots= num_shots+1
        
    
        #### center crop extraction
        if cur_w>=cur_h:
            start_pixel= int( np.floor( (cur_w - cur_h)/2 ) )
            cur_crop1= cur_key_frame1[start_pixel:start_pixel+cur_h,:,:]
            cur_crop2= cur_key_frame2[start_pixel:start_pixel+cur_h,:,:]
            cur_crop3= cur_key_frame3[start_pixel:start_pixel+cur_h,:,:]

        else:
            start_pixel= int( np.floor( (cur_h - cur_w)/2 ) )            
            cur_crop1= cur_key_frame1[:,start_pixel:start_pixel+cur_w,:]
            cur_crop2= cur_key_frame2[:,start_pixel:start_pixel+cur_w,:]
            cur_crop3= cur_key_frame3[:,start_pixel:start_pixel+cur_w,:]

            
        cur_crop_resized1= cv2.resize(cur_crop1, (112, 112))   
        cur_crop_resized2= cv2.resize(cur_crop2, (112, 112))  
        cur_crop_resized3= cv2.resize(cur_crop3, (112, 112)) 
        Vid_keyframes.append(cur_crop_resized1)  
        Vid_keyframes.append(cur_crop_resized2) 
        Vid_keyframes.append(cur_crop_resized3)
        num_ad_keyframes= num_ad_keyframes+3    
      

pickle.dump( Vid_keyframes, open( "Vid500_10shots_3keyframes.p", "wb" ), protocol= 2 )  
#pickle.dump( Ads_keyframes, open( "Ads_keyframes1.p", "wb" ), protocol= pickle.HIGHEST_PROTOCOL )        


end_time= time.time()
tot_time= end_time - st_time
print("total time=", tot_time)





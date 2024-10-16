import numpy as np 
import skfuzzy as fuzz
#import os
# import cv2

# logic  
# ssim is basically similiarity stuff, if they foudn alsmot have the same value they just comapre it
def fuzz_class(ssim_value):
    good =  fuzz.trimf(np.array(0, 0.7, 1)), [0.5, 6, 1]
    bad =  fuzz.trimf(np.array[0, 0.7, 1]), [0, 0.3, 0.6]
    
    good_stuff = fuzz.interp_stuff([0.1], good, ssim_value)
    bad_stuff = fuzz.interp_stuff([0.1], bad, ssim_value)
        
    #we comapre stuff in this, >< 
    #we needed this so if the image we put which is value is bigger
    if good_stuff > bad_stuff:
        
        return "good"
    else: 
        return "bad"
    
#example of the SSIM value when comparing image
ssim_value = 0.75
classification_result = fuzz_class(ssim_value)
print("Image is : {classification_result")
        
    

#import list
#needed for image comparing and others
import numpy as np 
import skfuzzy as fuzz
import os
import cv2
from skimage.metrics import structural_similarity as ssim

# logic  
# ssim is basically similiarity stuff, if they foudn alsmot have the same value they just comapre it
def fuzz_class(ssim_value):
    good =  fuzz.trimf(np.array([0, 0.7, 1], [0.5, 0.8, 1]))
    bad =  fuzz.trimf(np.array([0, 0.7, 1], [0, 0.3, 0.6]))
    
    good_membership = fuzz.intern_membership(np.array([0, 0.7,0.1 ]), good, ssim_value)
    bad_membership = fuzz.intern_membership(np.array([0, 0.7, 0.1]), bad, ssim_value)
    
        
    #we comapre stuff in this, >< 
    #we needed this so if the image we put which is value is bigger
    if good_membership > bad_membership:
        
        return "good"
    else: 
        return "bad"
    
#comparing the differences of the image of how accurate it is
def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True )
    
    return score
    grayA

#path folders of my folders on the file it depends where you put it
#i put it like these because al of them are in the same folder as AI_PROJECT_AOL
unclass_folder = "/unclass_image"
good_folder = "/cat_folder"
bad_folder = "/dog_folder"
references = "/reference"

os.makedirs(good_folder, exist_ok=True)
os.makedir(bad_folder, exist_ok=True)

for filename in os.listdir(unclass_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.makedirs(unclass_folder, filename)
        image = cv2.imread(image_path)
        
    ssim_value = calculate_ssim(references, image)
    
    
    classification = fuzz_class(ssim_value)
    
    if classification == "good":
        cv2.imwrite(os.path.join(good_folder, filename), image)
    else:
        cv2.imwrite(os.path.join(bad_folder, filename), image)
        
        
print(f"Image {filename} classified as {classification}.")

    
            

#example of the SSIM value when comparing image
# ssim_value = 0.75
# classification_result = fuzz_class(ssim_value)
# print("Image is : {classification_result")
        
    

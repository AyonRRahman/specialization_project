import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import product

def match_with_orb_config(img1, img2, nfeatures,scaleFactor, nlevels,find_good_match=True):
        
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures,scaleFactor=scaleFactor,nlevels=nlevels ,fastThreshold=7)

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des2 is None or des1 is None:
        print('init level failed')
        orb = cv2.ORB_create(nfeatures=nfeatures,scaleFactor=scaleFactor,nlevels=nlevels ,fastThreshold=3)

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
    
    if des2 is None or des1 is None:
        print('found None')
        # print(type(des2))
        return 0    

    # Use the BFMatcher to find the best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # print(f"lem{len(matches)}")
    # print(matches)
    # Apply ratio test
    if find_good_match:
        good_matches = []
    
        for match in matches:
            # Check if there are enough values to unpack
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    # else:
    #     good_matches=[]
    #     for match in matches:
    #         good_matches.append(1)
    
    # print(len(good_matches))
    
    return len(good_matches)


def match_images(img1, img2):
    # Load color images
    # print('match_images')
    nfeatures_list = [1000, 2000, 3000, 4000]
    scaleFactor_list = [1.2, 1.15, 1.10, 1.05]
    nlevels_list = [8,12,16,20]

    # Get all possible combinations
    combinations = list(product(nfeatures_list, scaleFactor_list, nlevels_list))
    match_dict = {}
    for nfeatures, scaleFactor, nlevels in combinations:
        # print(nfeatures, scaleFactor, nlevels)
        num_of_match = match_with_orb_config(img1, img2, nfeatures,scaleFactor, nlevels,find_good_match=True)
        match_dict[f'{nfeatures}_{scaleFactor}_{nlevels}'] = num_of_match

    return match_dict 

def find_total_matches(folder_path, timestamps, dataset_name):
    if dataset_name in os.listdir('orb_matching_results/'):
        return
        
    if dataset_name =='MH07':
        extention = 'corrected.png'
    else:
        extention='.png'
    total_matches = 0
    num_of_matches_list = []
    flag_first_image=0
    df = pd.DataFrame()

    for i in tqdm(range(len(timestamps) - 1)):
        if i==0:
            current_image = os.path.join(folder_path, timestamps[i])+extention
            next_image = os.path.join(folder_path, timestamps[i + 1])+extention
        else:
            current_image = next_image
            next_image = os.path.join(folder_path, timestamps[i + 1])+extention
            
        img1 = cv2.imread(current_image)
        img2 = cv2.imread(next_image)

        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        match_dict = match_images(img1, img2)
        
        new_row_series = pd.Series(match_dict, name=f'{i}')

        # Append the new row to the DataFrame
        df = df.append(new_row_series)
    
    df.to_csv(f'orb_matching_results/{dataset_name}.csv')
        
        # cv2.imshow('img1',img1)
        # cv2.imshow('img2',img2)
        # cv2.waitKey(10)

    # cv2.destroyAllWindows()
    
    


def evaluate_orb_config():
    time_stamps = "/home/ayon/git/orbslam3_docker/Datasets/EuRoC/time_stamp.txt"

    df = pd.read_csv(time_stamps, header=None)
    time_stamps = df[0].to_numpy().astype('str')

    for i in range(3,5):
        # print(i)
        print(f"matching for MH0{i}\n")
        image_folder = f"/home/ayon/git/orbslam3_docker/Datasets/EuRoC/MH0{i}/mav0/cam0/data"
        assert len(os.listdir(image_folder))==len(time_stamps)
        start_time = time.time()
        find_total_matches(image_folder, time_stamps,dataset_name = f'MH0{i}')
        end_time = time.time()
        
        

        
if __name__=='__main__':
    evaluate_orb_config()
    
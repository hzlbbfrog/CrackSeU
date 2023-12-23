"""
Author: Zhili He
For: Count the number of cracks in a given binary image
"""

import cv2
import numpy as np
from PIL import Image


def Binary_loader(Image_path):
    image = Image.open(Image_path)
    image = image.convert('L')
    image = np.array(image)
    return(image)


def Search_the_whole_crack(i, j, Total_number):
    
    Delta_x = [-1, -1, -1, 0, 0, 1, 1, 1] # 8 directions
    Delta_y = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    List = []
    List.append((i, j))
    Number_array[i][j] = Total_number
    Begin, End = 0, 0
    
    while(1):
        Now_x = List[Begin][0]
        Now_y = List[Begin][1]
        
        for i in range(8):
            Neibhor_x = Now_x + Delta_x[i]
            Neibhor_y = Now_y + Delta_y[i]
            if (Neibhor_x >= 0) and (Neibhor_x < H) and (Neibhor_y >=0) and (Neibhor_y < W):
                if (Mask[Neibhor_x][Neibhor_y] == Pixel_value) and (not Number_array[Neibhor_x][Neibhor_y]):
                    End += 1
                    List.append((Neibhor_x, Neibhor_y))
                    Number_array[Neibhor_x][Neibhor_y] = Total_number
        
        Begin += 1
        if (Begin > End):
            break


if __name__ =="__main__":
    
    Pixel_value = 255 # Pixel value = 255 or 0
    
    Dir_path='E:/Folder'
    Mask_name = '/Image_name.png'
    Mask_path =  Dir_path + Mask_name
    
    Mask = Binary_loader(Mask_path) # H and W
    H = Mask.shape[0]
    W = Mask.shape[1]
    
    Total_number = 0
    Number_array = np.zeros((H,W), dtype=np.int32)
    
    for i in range(H):
        for j in range(W):
            if (Mask[i][j] == Pixel_value) and (not Number_array[i][j]):
                Total_number += 1
                Search_the_whole_crack(i, j, Total_number)
                    
    print(Total_number)      
import torch as tr
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
import torch.nn.functional as F


class CNN_filter:
    def __init__(self,file):
        self.file = file
        self.to_tensor = tv.transforms.ToTensor()
        self.to_image = tv.transforms.ToPILImage()
        self.gray = self.read_image()
        self.image_tensor = self.to_tensor(self.gray)
        self.combinded = self.combining_filters()

    def read_image(self): # functuon for gray scaling of the image
        image = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE) # leser bildet direkt som gray scale
        return image

    #def to_tensor(self):
        #image_tensor = tr.tensor(self.gray, dtype = tr.float32).unsqueeze(0).unsqueeze(0)
        #image_tensor = image_tensor/255 # Normalise

       #return image_tensor    

    def filter(self):
    # Applying both verticaly and horrisontaly filters for the egdes
    # Taking the vertical filter from 2D to 4D whit .unsqueeze(0).unsqueeze(0)
        v_filter = tr.tensor([
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]],
                dtype = tr.float32
            ).unsqueeze(0).unsqueeze(0)
        
    # Taking the horrisontal filter from 2D to 4D whit .unsqueeze(0).unsqueeze(0)
        h_filter = tr.tensor([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]],
                dtype = tr.float32
            ).unsqueeze(0).unsqueeze(0)
        
        both_filters = tr.cat([v_filter, h_filter], dim=0)

        return both_filters
    
    def combining_filters(self):
        
        filters = self.filter()

        filter_output = F.conv2d(self.image_tensor,filters, padding=1)

        vert_out, horr_out = filter_output.squeeze(0)

        combinded = vert_out - horr_out # Dette funker jævlig bra!!
        #combinded = vert_out + horr_out
        #combinded = vert_out * horr_out
        #combinded = horr_out - vert_out
        

        return combinded
    
    def filtered_image(self):

        combinded = self.combining_filters()

        combinded_np = combinded.numpy()

        # Need to normalize so that open cv can understand the pixel valuse and konverting to uint8 format that is sutibel for open cv

        combinded_np = (combinded_np - combinded_np.min()) / (combinded_np.max() - combinded_np.min()) 
        combinded_np = (combinded_np * 255).astype(np.uint8) 

        return combinded_np

    #combinded = tr.sqrt(vert_out**2 + horr_out**2)
    #combinded = tr.abs(vert_out) + tr.abs(horr_out)
    #combinded = (combinded - combinded.min()) / (combinded.max() - combinded.min())


path_detected_sudoku = "./Sudoku_detected.jpg"

fil_cnn = CNN_filter(path_detected_sudoku).filtered_image()
cv2.imwrite("CNN_edge_detection_4.jpg", fil_cnn)

import torch
import random

class RandomOutline(object):
    def __init__(self, p):
        self.p = p
        self.dim = 3

    def __call__(self, img):
        h = img.size()[1]
        w = img.size()[2]
        
        h_size = int(h * self.p) # max h outline size
        w_size = int(w * self.p) # max w outline size

        for j in range(h):
            top = random.randint(0, h_size)
            bottom = random.randint(0, h_size)
            img[:, :top, j] = torch.from_numpy(np.random.rand(3,top))
            img[:, h-bottom:, j] = torch.from_numpy(np.random.rand(3,bottom))
            
        for j in range(w):
            left = random.randint(0, w_size)
            right = random.randint(0, w_size)
            img[:, j, :left] = torch.from_numpy(np.random.rand(3,left))
            img[:, j, w-right:] = torch.from_numpy(np.random.rand(3,right))
        
        return img
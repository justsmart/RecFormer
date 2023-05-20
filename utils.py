import numpy as np
import logging
import matplotlib.pyplot as plt
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.vals.append(val)
        self.sum = np.sum(self.vals)
        self.count = len(self.vals)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals)
        self.min = min(self.vals)
        self.min_ind = self.vals.index(self.min)
        self.max = max(self.vals)
        self.max_ind = self.vals.index(self.max)

def setLogger(logfile):
    logger = logging.getLogger()
    
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    
    
    while logger.handlers:
        logger.handlers.pop()
    if logfile:
        handler = logging.FileHandler(logfile,mode='w') 
        logger.addHandler(handler)
    logger.addHandler(console)
    return logger
def saveSingleImg(img,name,dpi):
    fig = plt.figure()
    plt.imshow(img, cmap='Greys_r')
    plt.axis('off')
    plt.savefig(name,dpi=dpi)
def saveImg(imo_mis,imo_exi,imr_mis,imr_exi,name):
    fig = plt.figure(figsize=(12,12))
    a = fig.add_subplot(2,2,1)    
    plt.imshow(imo_mis, cmap='Greys_r')
    a.set_title('miss before')
    b = fig.add_subplot(2,2,2)
    plt.imshow(imo_exi, cmap='Greys_r')
    b.set_title('exist before')
    c = fig.add_subplot(2,2,3)
    plt.imshow(imr_mis, cmap='Greys_r')
    c.set_title('miss recover')
    d = fig.add_subplot(2,2,4)
    plt.imshow(imr_exi, cmap='Greys_r')
    d.set_title('exist recover')
    plt.savefig(name)
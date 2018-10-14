import os

import torch
from torch.autograd import Variable
from datetime import datetime, timedelta

#import batcher
#from batcher import Batcher
import models
from models import ArcBinaryClassifier
from skimage import io
import numpy as np
import fnmatch
import scipy
from scipy import signal
from scipy.io import wavfile
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage

cuda=True
batchSize=64
imageSize=28
glimpseSize=8
numStates=9
numGlimpses=16
lr=0.002
name=None
load=None

pathpng = '/home/alekh/Desktop/CLIC/MNIST/'
pathjpeg = '/home/alekh/Desktop/resizedJPEG/'
quality = 60

to_tensor = ToTensor()
to_PIL = ToPILImage()

def load_data(batchsize):
    arr = np.random.randint(1,1000,batchsize)
    X = Variable(torch.ones(batchsize,1,28,28))
    #Y = Variable(torch.ones(batchsize,300,300,3))
    
    for i,a in enumerate(arr):
        imgpng = Image.open(pathpng+str(a)+".png").convert('LA')
        #imgjpeg = Image.open(pathjpeg+str(a)+".jpeg")
        #print(imgpng.size)
        imgpng = to_tensor(imgpng)[0,:,:]
        #imgjpeg = np.array(imgjpeg)
        #print(imgpng)
        X[i] = imgpng.float()
        #Y[i] = torch.from_numpy(imgjpeg).float()
        
    return X
    
def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def save_img(X,j):
    #img_count = X.shape[0]
    for i in range(1):
        img = X[i] * 255
        img = img.type(torch.ByteTensor).unsqueeze(2)
        img = to_PIL(img.data.cpu().numpy())
        img.save("temp"+str(j)+".PNG",quality=quality)

if cuda:
    #batcher.use_cuda = True
    models.use_cuda = True

if name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    name = "{}_{}_{}_{}".format(numGlimpses, glimpseSize, numStates,
                                    "cuda" if cuda else "cpu")

print("Will start training {} \n".format(name))

# make directory for storing models.
models_path = os.path.join("saved_models", name)
os.makedirs(models_path, exist_ok=True)

# initialise the model
discriminator = ArcBinaryClassifier(num_glimpses=numGlimpses,
                                    glimpse_h=glimpseSize,
                                    glimpse_w=glimpseSize,
                                    controller_out=numStates)

if cuda:
    discriminator.cuda()

# load from a previous checkpoint, if specified.
if load is not None:
    discriminator.load_state_dict(torch.load(os.path.join(models_path, load)))

# set up the optimizer.
mse = torch.nn.MSELoss(size_average=False)
if cuda:
    mse = mse.cuda()

optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

# load the dataset in memory.
#loader = Batcher(batch_size=batchSize, image_size=imageSize)

# ready to train ...
best_validation_loss = None
saving_threshold = 1.02
last_saved = datetime.utcnow()
save_every = timedelta(minutes=10)

i = -1

while True:
    i += 1

    X = load_data(batchSize)
    
    X = X.cuda()
    x_size = X.shape
    noise = Variable(torch.normal(means=torch.zeros(x_size),std=0.003).cuda())
    X_clean = X
    X = X + noise
    pred = discriminator(X_clean)
    loss = mse(pred, X_clean.float())

    if i % 100 == 0:
        save_img(pred,i)
        save_img(X_clean.squeeze(),str(i)+"orig")
        # validate your model
        X_val = load_data(batchSize)
        X_val = X_val.cuda()
        pred_val = discriminator(X_val)
        
        loss_val = mse(pred_val, X_val.float())

        training_loss = loss.data[0]
        validation_loss = loss_val.data[0]

        print("Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}".format(
            i, mse_loss(pred, X), training_loss, mse_loss(pred_val, X_val), validation_loss
        ))

        if best_validation_loss is None:
            best_validation_loss = validation_loss

        if best_validation_loss > (saving_threshold * validation_loss):
            print("Significantly improved validation loss from {} --> {}. Saving...".format(
                best_validation_loss, validation_loss
            ))
            discriminator.save_to_file(os.path.join(models_path, str(validation_loss)))
            best_validation_loss = validation_loss
            last_saved = datetime.utcnow()

        if last_saved + save_every < datetime.utcnow():
            print("It's been too long since we last saved the model. Saving...")
            discriminator.save_to_file(os.path.join(models_path, str(validation_loss)))
            last_saved = datetime.utcnow()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torchvision.transforms import ToTensor,ToPILImage
from skimage.draw import polygon_perimeter
from skimage import io
import numpy as np
use_cuda = True
to_PIL = ToPILImage()

class GlimpseWindow:
    """
    Generates glimpses from images using Cauchy kernels.

    Args:
        glimpse_h (int): The height of the glimpses to be generated.
        glimpse_w (int): The width of the glimpses to be generated.

    """

    def __init__(self, glimpse_h: int, glimpse_w: int):
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w

    @staticmethod
    def _get_filterbanks(delta_caps: Variable, center_caps: Variable, image_size: int, glimpse_size: int) -> Variable:
        """
        Generates Cauchy Filter Banks along a dimension.

        Args:
            delta_caps (B,):  A batch of deltas [-1, 1]
            center_caps (B,): A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            image_size (int): size of images along that dimension
            glimpse_size (int): size of glimpses to be generated along that dimension

        Returns:
            (B, image_size, glimpse_size): A batch of filter banks

        """

        # convert dimension sizes to float. lots of math ahead.
        image_size = float(image_size)
        glimpse_size = float(glimpse_size)

        # scale the centers and the deltas to map to the actual size of given image.
        centers = (image_size - 1) * (center_caps + 1) / 2.0  # (B)
        deltas = (float(image_size) / glimpse_size) * (1.0 - torch.abs(delta_caps))
        

        # calculate gamma for cauchy kernel
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))  # (B)

        # coordinate of pixels on the glimpse
        glimpse_pixels = Variable(torch.arange(0, glimpse_size) - (glimpse_size - 1.0) / 2.0)  # (glimpse_size)
        if use_cuda:
            glimpse_pixels = glimpse_pixels.cuda()

        # space out with delta
        glimpse_pixels = deltas[:, None] * glimpse_pixels[None, :]  # (B, glimpse_size)
        # center around the centers
        glimpse_pixels = centers[:, None] + glimpse_pixels  # (B, glimpse_size)

        # coordinates of pixels on the image
        image_pixels = Variable(torch.arange(0, image_size))  # (image_size)
        if use_cuda:
            image_pixels = image_pixels.cuda()

        fx = image_pixels - glimpse_pixels[:, :, None]  # (B, glimpse_size, image_size)
        fx = fx / gammas[:, None, None]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, None, None] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 1e-4)[:, :, None]  # we add a small constant in the denominator division by 0.

        return fx.transpose(1, 2)

    def get_attention_mask(self, glimpse_params: Variable, mask_h: int, mask_w: int) -> Variable:
        """
        For visualization, generate a heat map (or mask) of which pixels got the most "attention".

        Args:
            glimpse_params (B, hx):  A batch of glimpse parameters.
            mask_h (int): The height of the image for which the mask is being generated.
            mask_w (int): The width of the image for which the mask is being generated.

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended pixels weighted more.

        """

        batch_size, _ = glimpse_params.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=mask_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=mask_w, glimpse_size=self.glimpse_w)

        # (B, glimpse_h, glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))

        # find the attention mask that lead to the glimpse.
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))

        # scale to between 0 and 1.0
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()

        return mask

    def get_glimpse(self, images: Variable, glimpse_params: Variable) -> Variable:
        """
        Generate glimpses given images and glimpse parameters. This is the main method of this class.

        The glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
        represents the relative position of the center of the glimpse on the image. delta determines
        the zoom factor of the glimpse.

        Args:
            images (B, h, w):  A batch of images
            glimpse_params (B, 3):  A batch of glimpse parameters (h_center, w_center, delta)

        Returns:
            (B, glimpse_h, glimpse_w): A batch of glimpses.

        """
        #print(images.size())
        batch_size, image_h, image_w = images.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=image_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=image_w, glimpse_size=self.glimpse_w)

        # F_h.T * images * F_w
        glimpses = images
        glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
        glimpses = torch.bmm(glimpses, F_w)

        return glimpses  # (B, glimpse_h, glimpse_w)


class ARC(nn.Module):
    """
    This class implements the Attentive Recurrent Comparators. This module has two main parts.

    1.) controller: The RNN module that takes as input glimpses from a pair of images and emits a hidden state.

    2.) glimpser: A Linear layer that takes the hidden state emitted by the controller and generates the glimpse
                    parameters. These glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
                    represents the relative position of the center of the glimpse on the image. delta determines
                    the zoom factor of the glimpse.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, controller_out: int=128) -> None:
        super().__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out
        self.iter = 0
        # main modules of ARC

        self.controller = nn.LSTMCell(input_size=(glimpse_h * glimpse_w), hidden_size=self.controller_out)
        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)

        # this will actually generate glimpses from images using the glimpse parameters.
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w)
        
        

    def forward(self, image_pairs: Variable) -> Variable:
        """
        The method calls the internal _forward() method which returns hidden states for all time steps. This i

        Args:
            image_pairs (B, 2, h, w):  A batch of pairs of images

        Returns:
            (B, controller_out): A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """

        # return only the last hidden state
        all_hidden = self._forward(image_pairs)  # (2*num_glimpses, B, controller_out)
        last_hidden = all_hidden[-1, :, :]  # (B, controller_out)

        #return last_hidden
        return all_hidden

    def _forward(self, image_pairs: Variable) -> Variable:
        """
        The main forward method of ARC. But it returns hidden state from all time steps (all glimpses) as opposed to
        just the last one. See the exposed forward() method.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (2*num_glimpses, B, controller_out) Hidden states from ALL time steps.

        """

        # convert to images to float.
        image_pairs = image_pairs.float()

        # calculate the batch size
        batch_size = image_pairs.size()[0]

        # an array for collecting hidden states from each time step.
        all_hidden = []

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)
        Cx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)

        if use_cuda:
            Hx, Cx = Hx.cuda(), Cx.cuda()
        self.iter +=1
        # take `num_glimpses` glimpses for both images, alternatingly.
        for turn in range(self.num_glimpses):
            # select image to show, alternate between the first and second image in the pair
            images_to_observe = image_pairs[:,0,:,:]  # (B, h, w)

            # choose a portion from image to glimpse using attention
            glimpse_params = torch.tanh(self.glimpser(Hx))  # (B, 3)  a batch of glimpse params (x, y, delta)
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.view(batch_size, -1)  # (B, glimpse_h * glimpse_w), one time-step

            # feed the glimpses and the previous hidden state to the LSTM.
            Hx, Cx = self.controller(flattened_glimpses, (Hx, Cx))  # (B, controller_out), (B, controller_out)
            
            if self.iter%50==0:
                    img = images_to_observe[0,:,:]*255
                    img = img.type(torch.ByteTensor).unsqueeze(2)
                    img = np.asarray(to_PIL(img.data.cpu().numpy()))
                    img.setflags(write=1)
                    img2 = img
                    #print(img.shape)
                    print(glimpse_params[0,0],glimpse_params[0,1],glimpse_params[0,2])
                    
                    #x = ((glimpse_params[0,0].data.cpu().numpy()[0]+1)/2)*27
                    #y = ((glimpse_params[0,1].data.cpu().numpy()[0]+1)/2)*27
                    #delta = (28/8)*(1-np.abs(glimpse_params[0,2].data.cpu().numpy()[0]))
                    
                    x_caps = float(glimpse_params[0,0].data.cpu().numpy()[0])
                    y_caps = float(glimpse_params[0,1].data.cpu().numpy()[0])
                    delta_caps = float(glimpse_params[0,2].data.cpu().numpy()[0])
                    image_size = 28.0
                    glimpse_size = 8.0
                    x = (image_size - 1) * (x_caps + 1) / 2.0
                    y = (image_size - 1) * (y_caps + 1) / 2.0
                    delta = (float(image_size) / glimpse_size) * (1.0 - np.abs(delta_caps))
                    
                    print(x,y,delta)
                    r = [y-delta,y-delta,y+delta,y+delta]
                    c = [x-delta,x+delta,x+delta,x-delta]
                    #print(r,c)
                    rr, cc = polygon_perimeter(r,c,shape=img.shape, clip=False)
                    img[rr,cc] = 255
                    #print("vsldnvjldsvnc")
                    #print(img)
                    
                    print("-------------------------")
                    #print(img)
                    try:
                        io.imsave("glimpses/"+str(self.iter)+"_1_"+str(turn)+".png",img)
                        
                        #io.imsave("steps/"+str(self.iter)+"_2_"+str(int((turn+1)/2))+".png",img2)
                        
                    except ValueError:
                        print("Img not in -1 to 1")
                        #print(img)
                        #print(img2)
            # append this hidden state to all states
            all_hidden.append(Hx)

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)

        # return a batch of all hidden states.
        return all_hidden


class ArcBinaryClassifier(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """
    
    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, controller_out: int = 128):
        super().__init__()
        self.arc = ARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            controller_out=controller_out)
        num_filters = 10
        kern_size=(4,4)
        self.iter = 0
        # two dense layers, which take the hidden state from the controller of ARC and
        # classify the images as belonging to the same class or not.
        #self.dense1 = nn.Linear(controller_out, 64)
        #self.dense2 = nn.Linear(64, 1)
        self.deconv0 = nn.ConvTranspose2d(in_channels=controller_out, out_channels=num_filters * 8,kernel_size=(4,4), bias=False)
        self.bn0 = nn.BatchNorm2d(num_filters * 8)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=num_filters * 8, out_channels=num_filters * 4,kernel_size=(6,6), bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters * 4)
        # 4 x 4 -> 8 x 8
        self.deconv2 = nn.ConvTranspose2d(in_channels=num_filters * 4, out_channels=num_filters * 2,kernel_size=(7,7), stride=1, padding=0, bias=False,)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)
        # 8 x 8 -> 16 x 16
        self.deconv3 = nn.ConvTranspose2d(in_channels=num_filters * 2, out_channels=num_filters,kernel_size=(7,7), stride=1, padding=0, bias=False,)
        self.bn3 = nn.BatchNorm2d(num_filters)
        # 16 x 16 -> 32 x 32
        self.deconv4 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=1,kernel_size=(8,8), stride=1, padding=0, bias=False,)
    def forward(self, image_pairs: Variable) -> Variable:
        #arc_out = self.arc(image_pairs).unsqueeze(2).unsqueeze(3)
        all_hidden = self.arc(image_pairs)
        all_hidden_first = all_hidden[:,0,:].unsqueeze(2).unsqueeze(3)
        #print(all_hidden.shape)
        arc_out = all_hidden[-1, :, :].unsqueeze(2).unsqueeze(3)
        #print(arc_out[0])
        arc_out = F.sigmoid(arc_out);
        all_hidden_first = F.sigmoid(all_hidden_first);
        #arc_out = arc_out.type(torch.FloatTensor).cuda()
        #print(arc_out.shape)
        x = F.relu(self.bn0(self.deconv0(arc_out)))
        y = F.relu(self.bn0(self.deconv0(all_hidden_first)))
        #print(x.shape)
        x = F.relu(self.bn1(self.deconv1(x)))
        y = F.relu(self.bn1(self.deconv1(y)))
        #print(x.shape)
        x = F.relu(self.bn2(self.deconv2(x)))
        y = F.relu(self.bn2(self.deconv2(y)))
        #print(x.shape)
        x = F.relu(self.bn3(self.deconv3(x)))
        y = F.relu(self.bn3(self.deconv3(y)))
        #print(x.shape)
        x = self.deconv4(x)
        y = self.deconv4(y)
        #print(x.shape)
        x = x.squeeze()         
        y = y.squeeze()
        #print(x.shape)
        images_gen = F.tanh(x)
        step_imgs = F.tanh(y)
        #print("----")
        self.iter+=1
        if(self.iter%500==0):
            for i in range(16):
                img = step_imgs[i] * 255
                img = img.type(torch.ByteTensor).unsqueeze(2)
                img = to_PIL(img.data.cpu().numpy())
                img.save("steps/temp"+str(self.iter)+"_"+str(i)+".PNG",quality=60)
        return images_gen

    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

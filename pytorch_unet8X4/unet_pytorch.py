
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tensorflow.compat.v1 as tfc
import tensorflow as tf
from PIL import Image
import os

device = None
if torch.cuda.is_available():
    # device = torch.cuda.get_device_name(1)  # you can continue going on here, like cuda:1 cuda:2....etc. 
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.                                                  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


ldct_train = np.load('/home/npovey/data/new_idea/sparseview_60_train_3600.npy') # loads saved array into variable sparseview_60_train.
ndct_train = np.load('/home/npovey/data/new_idea/ndct_train_3600.npy') # loads saved array into variable ndct_train.
ldct_test = np.load('/home/npovey/data/new_idea/sparseview_60_test_354.npy') # loads saved array into variable sparseview_60_test.
ndct_test = np.load('/home/npovey/data/new_idea/ndct_test_354.npy') # loads saved array into variable ndct_test.



# ------test data -----------
#X_test = torch.from_numpy(ldct_test).view(354, 1, 512, 512)
#y_test = torch.from_numpy(ndct_test).view(354, 1, 512, 512)
#print(X_test.shape)
#print(y_test.shape)
#X_test, y_test = X_test.to(device), y_test.to(device)
#----test data ends-------

# ------validation data --------
#ldct_vald = ldct_train[3560:3600,:,:,:] 
#ndct_vald = ndct_train[3560:3600,:,:,:] 
#X_vald = torch.from_numpy(ldct_vald).view(40, 1, 512, 512)
#y_vald = torch.from_numpy(ndct_vald).view(40, 1, 512, 512)
#print(X_vald.shape)
#print(y_vald.shape)
#X_vald, y_vald = X_vald.to(device), y_vald.to(device)
# ------end validation data ------



def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maxval = np.amax(im1)
    psnr = 10 * np.log10(maxval ** 2 / mse)
    return psnr

# Needed to get denoised images six at the time
# can't feed all images into a model.
# X_test: sparseview images
# y_test: clean image
def avg_psnr(model, X_vald, y_vald):
    model.eval()
    num_batches = int(30/10)
    # denoised_image = torch.empty(354,1,512, 512, dtype=torch.float)
    denoised_image = torch.randn(30,1,512, 512)
    with torch.no_grad():
        for idx in range(num_batches):
            model.zero_grad()
            denoised_image[idx*10:(idx+1)*10,:,:,:]= model(X_vald[idx*10:(idx+1)*10,:,:,:])

    ## find avg psnr
    psnr_sum = 0
    for i in range(len(X_vald)):
        psnr = cal_psnr(y_vald[i,:,:,:].cpu().data.numpy(), denoised_image[i,:,:,:].cpu().data.numpy())
        psnr_sum += psnr
    avg_psnr = psnr_sum / len(X_vald)
    return avg_psnr

# test 
def test(model):
    model.eval()

    X_test = torch.from_numpy(ldct_test).view(354, 1, 512, 512)
    y_test = torch.from_numpy(ndct_test).view(354, 1, 512, 512)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)
    X_test, y_test = X_test.to(device), y_test.to(device)

    num_batches = int(354/6)                                                                                                                       
    denoised_image = torch.empty(354,1,512, 512, dtype=torch.float)
    with torch.no_grad():
        for idx in range(num_batches):
            model.zero_grad()
            denoised_image[idx*6:(idx+1)*6,:,:,:]= model(X_test[idx*6:(idx+1)*6,:,:,:])
    psnr_sum = 0
    for i in range(len(X_test)):
        psnr = cal_psnr(y_test[i,:,:,:].cpu().data.numpy(), denoised_image[i,:,:,:].cpu().data.numpy())
        print("image: ",i ,"PSNR: " , psnr)  
        psnr_sum += psnr
    avg_psnr = psnr_sum / len(X_test)
    print("Avg PSNR: ",avg_psnr)
    
    # save images as .flt files                                                                                                                         
    save_dir = "/home/npovey/data/pytorch_unet8X4/test"
    rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=index)), 'wb') for index in range (354)]
    for index in range(354):
        img = np.asarray(denoised_image[index,:,:,:])
        img.tofile(rawfiles[index])

def save_png_images_0_1_12(model):
    model.eval()

    X_test = torch.from_numpy(ldct_test).view(354, 1, 512, 512)
    print(X_test.shape)
    X_test = X_test.to(device)
    denoised_image= model(X_test[0:2,:,:,:])
                                                                                                                                                           
    a = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(a)
    a = np.clip(255 * a/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((a).astype(np.uint8))
    result.save('pytorch_unet_0.png')
    
    
    b = denoised_image[1].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_1.png')
    
    denoised_image= model(X_test[12:16,:,:,:])    
    b = denoised_image[0].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_12.png')

    
    b = X_test[12].view(512,512).cpu().data.numpy()
    scalef = np.amax(b)
    b = np.clip(255 * b/scalef, 0, 255).astype('uint8')
    result = Image.fromarray((b).astype(np.uint8))
    result.save('pytorch_unet_12_ldct.png')



def denoise_all(model):
    print("denoising all")
    model.eval()
    for i in range(9):
        print("i: ",i)
        num_batches = int(400/4)
        ldct_train7 = ldct_train[(i*400):(i+1)*400,:,:,:]
        ldct_train7 = ldct_train7.reshape(400,1,512,512)
        X_train = torch.from_numpy(ldct_train7)
        X_train = X_train.to(device)
        
        denoised_image = torch.empty(400,1,512, 512, dtype=torch.float)
        with torch.no_grad():
            for idx in range(num_batches):
                # print("batch_number",idx)
                model.zero_grad()
                denoised_image[idx*4:(idx+1)*4,:,:,:]= model(X_train[idx*4:(idx+1)*4,:,:,:])
                # save images as .flt files                                                                                                                                                             
        save_dir = "/home/npovey/data/pytorch_unet8X4/denoised_images"
        rawfiles = [open(os.path.join(save_dir, "test_{num:08d}.flt".format(num=index+(i*400))), 'wb') for index in range (400)]
        for index in range(400):
            # print(index+(i*400))
            img = np.asarray(denoised_image[index,:,:,:])
            img.tofile(rawfiles[index])



# Unet model with all filters from UNET orgiginal paper
# Define model
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # print(1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=32)
        self.conv2 =  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=64)
        self.conv4 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(num_features=128)
        self.conv6 =  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # print(4)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch7 = nn.BatchNorm2d(num_features=256)
        self.conv8 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch8 = nn.BatchNorm2d(num_features=256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        
        # print(5)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batch9 = nn.BatchNorm2d(num_features=512)
        self.conv10 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batch10 = nn.BatchNorm2d(num_features=512)

        # print(6)
        self.trans1 = nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be add]
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.batch11 = nn.BatchNorm2d(num_features=256)
        self.conv12 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.batch12 = nn.BatchNorm2d(num_features=256)

        # print(7)
        self.trans2 = nn.ConvTranspose2d(in_channels=256,out_channels=128, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.batch13 = nn.BatchNorm2d(num_features=128)
        self.conv14 =  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batch14 = nn.BatchNorm2d(num_features=128)
        
        # print(8)
        self.trans3 = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv15 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.batch15 = nn.BatchNorm2d(num_features=64)
        self.conv16 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batch16 = nn.BatchNorm2d(num_features=64)

        # print(9)
        self.trans4 = nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=(2, 2), stride=2, padding=0)
        ## concatenate [channels must be added]
        self.conv17 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.batch17 = nn.BatchNorm2d(num_features=32)
        self.conv18 =  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch18 = nn.BatchNorm2d(num_features=32)
        self.conv19 =  nn.Conv2d(32, out_channels=1, kernel_size=1, padding=0)

    def forward(self, inp):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # print("A")
        x = self.conv1(inp)
        x = self.batch1(x)
        x = F.relu(x)
        c1 = self.conv2(x)
        x = self.batch2(c1)
        x = F.relu(x)
        x = self.pool1(x)

        # print("B")
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        c2 = self.conv4(x)
        x = self.batch4(c2)
        x = F.relu(x)
        x = self.pool2(x)

        # print("C")
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.relu(x)
        c3 = self.conv6(x)
        x = self.batch6(c3)
        x = F.relu(x)
        x = self.pool3(x)

        # print("D")
        x = self.conv7(x)
        x = self.batch7(x)
        x = F.relu(x)
        c4 = self.conv8(x)
        x = self.batch8(c4)
        x = F.relu(x)
        x = self.pool4(x)

        # print("E")
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.conv10(x)
        x = self.batch10(x)

        # print("F")
        u1 = self.trans1(x)
        x = torch.cat((u1, c4),1)
        x = self.conv11(x)
        x = self.batch11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.batch12(x)
        x = F.relu(x)
        
        # print("G")
        u2 = self.trans2(x)
        x = torch.cat((u2, c3),1)
        x = self.conv13(x)
        x = self.batch13(x)
        x = F.relu(x)
        x = self.conv14(x)
        x = self.batch14(x)
        x = F.relu(x)

        # print("H")
        u3 = self.trans3(x)
        x = torch.cat((u3, c2),1)
        x = self.conv15(x)
        x = self.batch15(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = self.batch16(x)
        x = F.relu(x)

        # print("I")
        u4 = self.trans4(x)
        x = torch.cat((u4, c1),1)
        x = self.conv17(x)
        x = self.batch17(x)
        x = F.relu(x)
        x = self.conv18(x)
        x = self.batch18(x)
        x = F.relu(x)
        x = self.conv19(x)
        x = inp - x
        return x






def train(model):

    PATH = "/home/npovey/data/pytorch_unet8X4/unet_weights.pth"
    # -------validation data -----
    ldct_vald = ldct_train[3570:3600,:,:,:]
    ndct_vald = ndct_train[3570:3600,:,:,:]
    X_vald = torch.from_numpy(ldct_vald).view(30, 1, 512, 512)
    y_vald = torch.from_numpy(ndct_vald).view(30, 1, 512, 512)
    print("X_vald.shape", X_vald.shape)
    print("y_vald.shape", y_vald.shape)
    X_vald, y_vald = X_vald.to(device), y_vald.to(device)
    # -------validation data ------------




    start = time.time()
    model.train()
    n_epochs = 50
    batch_size = 8
    image_width = 512
    image_height = 512
    length = 1184
    losses = []
    psnrs = []
    #im = torch.randn(batch_size, im_channels, image_width, image_height)
    loss_func = nn.MSELoss()
    # optim = torch.optim.SGD(model.parameters(), lr=0.1,weight_decay=0.0001, momentum=0.99)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=0.0001, momentum=0.99)
    # optim = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=0.00001, momentum=0.99)

    print('iter,\tloss')
    z = int(length/batch_size)
    for epoch in range(n_epochs):
        # --- dividing learning reate in half every 10 epoch
        if(epoch < 10):
            optim = torch.optim.Adam(model.parameters(), lr=0.001)
        if(epoch < 20):
            optim = torch.optim.Adam(model.parameters(), lr=0.0005)
        if(epoch < 30):
            optim = torch.optim.Adam(model.parameters(), lr=0.00025)
        if(epoch < 40):
            optim = torch.optim.Adam(model.parameters(), lr=0.000125)
        if(epoch < 50):
            optim = torch.optim.Adam(model.parameters(), lr=0.0000635)


        print()
        print("Epoch",epoch)
        # iterate over all batches

        print("set[0:1200]")                                                                                                        
        ldct_train7 = ldct_train[0:1184,:,:,:]
        ndct_train7 = ndct_train[0:1184,:,:,:]
        p = np.random.permutation(1184)
        ldct_train7 = ldct_train7[p,:,:,:]
        ndct_train7 = ndct_train7[p,:,:,:]
        
        ldct_train2 = ldct_train7.reshape(1184,1,512,512)
        ndct_train2 = ndct_train7.reshape(1184,1,512,512)
        
        X_torch = torch.from_numpy(ldct_train2)
        y_torch = torch.from_numpy(ndct_train2)
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)

        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss) 
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            

        # -----------------flipud augment images ------------------                                                                    
        print("image augmentation 1")
          
        flipped_l = np.flipud(ldct_train7)
        flipped_m = np.flipud(ndct_train7)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            #optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            


        # -----------------rotate 90 degrees augment images ------------------                                                         
        print("image augmentation 2")

        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))



        # -----------------rotate 90+flip degrees augment images ------------------                                                   
        print("image augmentation 3")

        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
                #                print("working on batches 1 trough 75...")

        # -----------------rotate 180 degrees augment images ------------------                      
        print("image augmentation 4")

        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):        
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))



        # -----------------rotate 180 + flip degrees augment images ------------------                                                 
        print("image augmentation 5")
         
        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))


        # -----------------rotate 270 degrees augment images ------------------                                                        
        print("image augmentation 6")

        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # -----------------rotate 270 + flip degrees augment images ------------------                                                 
        print("image augmentation 7")

        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))



        ##-------------- end augmented    
        # ---- set from 1200 to 2400 -----
        print("set[1200:2400]")
        ldct_train7 = ldct_train[1184:2368,:,:,:]
        ndct_train7 = ndct_train[1184:2368,:,:,:]
        p = np.random.permutation(1184)
        ldct_train7 = ldct_train7[p,:,:,:]
        ndct_train7 = ndct_train7[p,:,:,:]
        
        ldct_train2 = ldct_train7.reshape(1184,1,512,512)
        ndct_train2 = ndct_train7.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(ldct_train2)
        y_torch = torch.from_numpy(ndct_train2)
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # augment images ------------------
        print("image augmentation 1")
        flipped_l = np.flipud(ldct_train7)
        flipped_m = np.flipud(ndct_train7)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            ##-------------- end augmented  
            
        # -----------------rotate 90 degrees augment images ------------------
        print("image augmentation 2") 
        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            

        # -----------------rotate 90 + flip degrees augment images ------------------
        print("image augmentation 3") 
        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))


        # -----------------rotate 180 degrees augment images ------------------                                                      
        print("image augmentation 4")
        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))


        # -----------------rotate 180 + flip degrees augment images ------------------                                               
        print("image augmentation 5")

        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # -----------------rotate 270 degrees augment images ------------------                                                       
        print("image augmentation 6")

        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):        
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            
        # -----------------rotate 270 + flip degrees augment images ------------------
        print("image augmentation 7")

        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
        ##-------------- end augmented      
        # ---- end from 1200 t0 2400 ----      
        # ---- set from 2400 to 3600 ------ 
        print("set[2400:3600]")                                                                                                     
        ldct_train7 = ldct_train[2386:3570,:,:,:]
        ndct_train7 = ndct_train[2386:3570,:,:,:]
        p = np.random.permutation(1184)
        ldct_train7 = ldct_train7[p,:,:,:]
        ndct_train7 = ndct_train7[p,:,:,:]
        
        ldct_train2 = ldct_train7.reshape(1184,1,512,512)
        ndct_train2 = ndct_train7.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(ldct_train2)
        y_torch = torch.from_numpy(ndct_train2)
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        z2 = int(1184/32)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss) 
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # augment images ------------------                                                                                        
        print("image augmentation 1")
        flipped_l = np.flipud(ldct_train7)
        flipped_m = np.flipud(ndct_train7)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # -----------------rotate 90 degrees augment images ------------------                                                      
        print("image augmentation 2")
        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
                #               print("working on batches 1 trough 75...")
                
                
        # -----------------rotate 90 + flip degrees augment images ------------------                                               
        print("image augmentation 3")

        flipped_l = np.rot90(ldct_train7, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
           
        # -----------------rotate 180 degrees augment images ------------------ 
        print("image augmentation 4")

        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):            
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # -----------------rotate 180 + flip degrees augment images ------------------                                                 
        print("image augmentation 5") 
        flipped_l = np.rot90(ldct_train7, k=2, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7, k=2, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))


        # -----------------rotate 270 degrees augment images ------------------                                                        
        print("image augmentation 6")

        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))

        # -----------------rotate 270 + flip degrees augment images ------------------                                                 
        print("image augmentation 7")
         
        flipped_l = np.rot90(ldct_train7,k=3, axes=(-2,-1))
        flipped_m = np.rot90(ndct_train7,k=3, axes=(-2,-1))
        flipped_l = np.flipud( flipped_l)
        flipped_m = np.flipud( flipped_m)
        flipped_l = flipped_l.reshape(1184,1,512,512)
        flipped_m = flipped_m.reshape(1184,1,512,512)
        X_torch = torch.from_numpy(flipped_l.copy())
        y_torch = torch.from_numpy(flipped_m.copy())
        X_torch, y_torch = X_torch.to(device), y_torch.to(device)
        for i in range(z2):        
            optim.zero_grad()
            y_hat = model(X_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            loss = loss_func(y_hat, y_torch[i*batch_size:(i+1)*batch_size,:,:,:])
            losses.append(loss)
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print('batch: {},\t{:.7f}'.format(i, loss.item()))
            
            ##-------------- end augmented  

            # set from 2400 to 3600 --------------------------

        # save model every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        }, PATH)
            
        psnr = avg_psnr(model, X_vald, y_vald)
        psnrs.append(psnr)
        model.train()
        print('epoch: {},\tAvg PSNR {:.7f}'.format(epoch, psnr))
        torch.cuda.empty_cache()
        
    np.save('/home/npovey/data/pytorch_unet8X4/losses', losses)
    np.save('/home/npovey/data/pytorch_unet8X4/psnrs', psnrs)
    end = time.time()
    print("training time for 1 epoch: ",end - start)

# ---------main -------------
#model = Unet()
#model.to(device)
PATH = "/home/npovey/data/pytorch_unet8X4/unet_weights.pth"
model = Unet()
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

#model = Unet()
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optim.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']               
#loss = checkpoint['loss']                                                                                                                                                                                 
#model.to(device)




train(model)
test(model)
torch.cuda.empty_cache()
denoise_all(model)
torch.cuda.empty_cache()
save_png_images_0_1_12(model)


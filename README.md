**Applying U-Net Model to Improve the Quality of Low Dose and Sparse-View CT Scans**

**Overview**

There are two ways of reducing the radiation dose to the patient while taking CT scans: (i) acquiring data with a lower beam intensity, and (ii) reducing the number of views (sparse-view CT). Both methods decrease image quality, but in different ways. The low dose image looks like a noisy version of a normal image, while the sparse-view CT has streaking artifacts. Our interest is to improve the quality of low dose and sparse-view CT images by using deep neural networks. The approach is based on the well-known U-Net model [1] implemented using the Keras functional API with the TensorFlow backend. We compared the results with the Denoising CNN model that has previously been described [2].

**Introduction**

Due to concerns with radiation exposure while taking CT scans, researchers are trying to find ways to reduce the dose of radiation with minimum impact on image quality. We investigated denoising for 3 different types of low dose images and 3 different types of sparse-view images.

We took low dose and sparse-view images and reconstructed them using a U-Net-based denoising model. We compared this with our previous denoising model which was CNN-based. 

The CT-images were obtained from the Cancer Imaging Archive’s QIN LUNG CT dataset. Altogether we had 3954 slices from a total of 47 patient studies. For each image type we used 3600 images to train and 354 images to test. All images are of size 512x512.

**U-Net Model vs. Denoising CNN Model**


**U-Net model [1]**


![unet](/formulas/unet.png)






The U-Net model was trained on full images of size 512x512.
The training process took approximately 12 hours. 
Total parameters in the network: 487,921

**Denoising CNN model [3]**

![denoisingCNN](/formulas/denoisingCNN.png)


(i) First layer has Conv+ ReLU with filter 3x3x64.
(ii) Second through fifteenth layers has Conv+ ReLU + BN with filters 3x3x64.
(iii) Last layer only has Conv with 3x3x64 filters. 
Subtract found residual image from the noisy image.
Denoising CNN model trains on cropped images of size 32x32. 

The training process took approximately 14 hours.

**Results**

The average PSNR and SSIM values over 354 test images are displayed in the table below. 

| Low Dose Image | UNet                              | DnCNN                             |
| -------------- | ----------------------------------| ----------------------------------|
| sparseview_60  | Avg PSNR: 33.28	Avg SSIM: 0.8858 | Avg PSNR: 32.30  Avg SSIM: 0.8560 |
| sparseview_90  | Avg PSNR: 35.42	Avg SSIM: 0.9038 | Avg PSNR: 35.13  Avg SSIM: 0.8892 |
| sparseview_180 | Avg PSNR: 39.48	Avg SSIM: 0.9319 | Avg PSNR: 39.77  Avg SSIM: 0.9341 |
| ldct_7e4       | Avg PSNR: 41.78	Avg SSIM: 0.9429 | Avg PSNR: 42.00	Avg SSIM: 0.9444 |
| ldct_1e5       | Avg PSNR: 42.11	Avg SSIM: 0.9441 | Avg PSNR: 42.32	Avg SSIM: 0.9456 |
| ldct_2e5       | Avg PSNR: 42.69	Avg SSIM: 0.9466 | Avg PSNR: 42.87	Avg SSIM: 0.9477 |

![psnr](/formulas/psnr.png)



![ssim](/formulas/ssim.png)



**Conclusion**

The U-Net model was comparable to the Denoising CNN model under most conditions, but U-Net provides better results for the sparse-view at the lowest sampling rate, where the number of views is reduced to 60 and the artifacts are most severe. The improvement with respect to both PSNR and SSIM was roughly 3%. The U-Net was also slightly faster to train.

**References and Acknowledgments**

[1] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. MICCAI. 2015

[2] S. Coulter M. Simms T. Humphries, D. Si and R. Xing. Comparison of deep learning approaches to low dose CT using low intensity and sparse view data. SPIE Medical Imaging. 2019

[3] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. Beyond a gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Trans. Imag. Proc. 2016.

[4] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P.  Simoncelli. "Image quality assessment: from error visibility to structural similarity". IEEE Trans. Imag. Proc. 2004

---------------------------------------------------------------------------------------------------------------------------

This version uses new unet model (keras implementation with batch normalization) and does image augmentation without cropping.  After reading a post from Andrej Karpathy about big mistakes people do in ML I decided to improve on unet14 by adding   bias=False while training. One of the mistakes he is writing is "5) you didn't use bias=False for your Linear/Conv2d layer when using BatchNorm". 

<https://twitter.com/karpathy/status/1013245864570073090?lang=en>

The performnce imporved slightly after adding bias=False.

Need the following environment to run the code.

```{python}
(base) [npovey@ka ~]$ conda create -n keras-gpu python=3.6 numpy scipy keras-gpu
(base) [npovey@ka unet4]$ conda activate keras-gpu
(keras-gpu) [npovey@ka unet4]$ pip install pandas
(keras-gpu) [npovey@ka unet4]$ pip install Pillow
(keras-gpu) [npovey@ka unet4]$ pip install matplotlib
(keras-gpu) [npovey@ka unet4]$ python3 main.py 


```

other useful commands

```{
export CUDA_VISIBLE_DEVICES=0
(keras-gpu) [npovey@ka unet14]$ kill -KILL 43377
(keras-gpu) [npovey@ka unet14]$ screen -S unet148
(press ctlr+A, D to detach from screen)
[detached from 50566.unet148]
perl -pi -e 's/\Q$_// if ($. == 14);' ~/.ssh/known_hosts

(base) [npovey@ka dncnn1]$ source activate keras-gpu
(keras-gpu) [npovey@ka dncnn1]$ python main.py > output_dncnn1_60.txt
scp -r npovey@ka:/data/CT_data/images/* CT_data/images
```

##### View a Tensorboard

```py
tensorboard --logdir=./logs
http://0.0.0.0:6006
```
##### UNet vs DnCNN (Denoising CNN) training time

| Low Dose Image | UNet                              | DnCNN                             |
| -------------- | ----------------------------------| ----------------------------------|
| sparseview_60  | Avg PSNR: 33.28	Avg SSIM: 0.8858 | Avg PSNR: 32.30  Avg SSIM: 0.8560 |
| sparseview_90  | Avg PSNR: 35.42	Avg SSIM: 0.9038 | Avg PSNR: 35.13  Avg SSIM: 0.8892 |
| sparseview_180 | Avg PSNR: 39.48	Avg SSIM: 0.9319 | Avg PSNR: 39.77  Avg SSIM: 0.9341 |
| ldct_7e4       | Avg PSNR: 41.78	Avg SSIM: 0.9429 | Avg PSNR: 42.00	Avg SSIM: 0.9444 |
| ldct_1e5       | Avg PSNR: 42.11	Avg SSIM: 0.9441 | Avg PSNR: 42.32	Avg SSIM: 0.9456 |
| ldct_2e5       | Avg PSNR: 42.69	Avg SSIM: 0.9466 | Avg PSNR: 42.87	Avg SSIM: 0.9477 |


##### DnCnn vs UNet15 training time

| DnCnn                  | UNet                   |
| :--------------------- | ---------------------- |
| 14 hours for 50 epochs | 10 hours for 50 epochs |

##### UNet15 (with batch normalization and bias=False) 

| Low Dose Image | DnCnn                               | UNet15                              |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 33.28	Avg SSIM: 0.8858 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 35.42	Avg SSIM: 0.9038 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.48	Avg SSIM: 0.9319 |
| ldct_7e4       | Avg PSNR: 42.00	Avg SSIM: 0.9444 | Avg PSNR: 41.78	Avg SSIM: 0.9429 |
| ldct_1e5       | Avg PSNR: 42.32	Avg SSIM: 0.9456 | Avg PSNR: 42.11	Avg SSIM: 0.9441 |
| ldct_2e5       | Avg PSNR: 42.87	Avg SSIM: 0.9477 | Avg PSNR: 42.69	Avg SSIM: 0.9466 |

Old data link  ldct_1e5: Avg PSNR: 41.03	Avg SSIM: 0.9306

##### UNet14 (with batch normalization)

| Low Dose Image | DnCnn                               | UNet14(with batch normalization)    |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 33.12	Avg SSIM: 0.8823 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 35.45	Avg SSIM: 0.9043 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.28	Avg SSIM: 0.9303 |
| ldct_7e4       | Avg PSNR: 42.00	Avg SSIM: 0.9444 | Avg PSNR: 41.75	Avg SSIM: 0.9426 |
| ldct_1e5       | Avg PSNR: 42.32	Avg SSIM: 0.9456 | Avg PSNR: 42.16	Avg SSIM: 0.9444 |
| ldct_2e5       | Avg PSNR: 42.87	Avg SSIM: 0.9477 | Avg PSNR: 42.70	Avg SSIM: 0.9466 |



##### UNet6 results (no cropping, no batch normalization)

| Low Dose Image | DnCnn                               | UNet6(without batch normalization)  |
| -------------- | ----------------------------------- | ----------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | Avg PSNR: 33.26	Avg SSIM: 0.8851 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | Avg PSNR: 35.53	Avg SSIM: 0.9042 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | Avg PSNR: 39.44	Avg SSIM: 0.9315 |

##### Unet-keras3 results (2.5 hours training time, no cropping, no image augmentation, no bn)

| Low Dose Image | DnCnn                               | Unet-keras3                       |
| -------------- | ----------------------------------- | --------------------------------- |
| sparseview_60  | Avg PSNR: 32.30    Avg SSIM: 0.8560 | psnr 50 epochs: 37.71519761116282 |
| sparseview_90  | Avg PSNR: 35.13    Avg SSIM: 0.8892 | psnr 50 epochs: 39.68172444470224 |
| sparseview_180 | Avg PSNR: 39.77    Avg SSIM: 0.9341 | psnr 50 epochs:  42.8100640160669 |
| ldct_1e5       |                                     | psnr 50 epochs: 45.1429232130602  |
| ldct_2e5       |                                     |                                   |
| ldct_7e4       |                                     |                                   |

Unet

```{python}
inputs = Input((None, None,1))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)


c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)


u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

output_img = Conv2D(1, (1, 1)) (c9)
subtracted = Subtract()([inputs, output_img])


model = Model(inputs=[inputs], outputs=[subtracted])
model.compile(optimizer='adam', loss='mse', metrics=[tf_psnr])


```



```{python}
model.summary()
```

```{python}
Model: "model_1"
__________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==========================================================================================
input_1 (InputLayer)            (None, None, None, 1 0                                            
__________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, None, None, 8 80          input_1[0][0]                    
__________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, None, None, 8 584         conv2d_1[0][0]                   
__________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, None, None, 8 0           conv2d_2[0][0]                   
__________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, None, None, 1 1168        max_pooling2d_1[0][0]            
__________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, None, None, 1 2320        conv2d_3[0][0]                   
__________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, None, None, 1 0           conv2d_4[0][0]                   
__________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, None, None, 3 4640        max_pooling2d_2[0][0]            
__________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, None, None, 3 9248        conv2d_5[0][0]                   
__________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, None, None, 3 0           conv2d_6[0][0]                   
__________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, None, None, 6 18496       max_pooling2d_3[0][0]            
__________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, None, None, 6 36928       conv2d_7[0][0]                   
__________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, None, None, 6 0           conv2d_8[0][0]                   
__________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, None, None, 1 73856       max_pooling2d_4[0][0]            
__________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, None, None, 1 147584      conv2d_9[0][0]                   
__________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, None, None, 6 32832       conv2d_10[0][0]                  
__________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, None, None, 1 0           conv2d_transpose_1[0][0]         
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, None, None, 6 73792       concatenate_1[0][0]              
__________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, None, None, 6 36928       conv2d_11[0][0]                  
__________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, None, None, 3 8224        conv2d_12[0][0]                  
__________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, None, None, 6 0           conv2d_transpose_2[0][0]         
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, None, None, 3 18464       concatenate_2[0][0]              
__________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, None, None, 3 9248        conv2d_13[0][0]                  
__________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, None, None, 1 2064        conv2d_14[0][0]                  
__________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, None, None, 3 0           conv2d_transpose_3[0][0]         
                                                                 conv2d_4[0][0]                   
__________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, None, None, 1 4624        concatenate_3[0][0]              
__________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, None, None, 1 2320        conv2d_15[0][0]                  
__________________________________________________________________________________________
conv2d_transpose_4 (Conv2DTrans (None, None, None, 8 520         conv2d_16[0][0]                  
__________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, None, None, 1 0           conv2d_transpose_4[0][0]         
                                                                 conv2d_2[0][0]                   
__________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, None, None, 8 1160        concatenate_4[0][0]              
__________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, None, None, 8 584         conv2d_17[0][0]                  
__________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, None, None, 1 9           conv2d_18[0][0]                  
__________________________________________________________________________________________
subtract_1 (Subtract)           (None, None, None, 1 0           input_1[0][0]                    
                                                                 conv2d_19[0][0]                  
==========================================================================================
Total params: 485,673
Trainable params: 485,673
Non-trainable params: 0
```





```{python}
# UNet
inputs = Input((None, None,1))

c1 = Conv2D(8, (3, 3), padding='same') (inputs)
c1 = Activation('relu')(c1)
c1 = Conv2D(8, (3, 3), padding='same', use_bias=False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), padding='same', use_bias=False) (p1)
c2 = BatchNormalization()(c2)
c2 = Activation('relu')(c2)
c2 = Conv2D(16, (3, 3), padding='same', use_bias=False) (c2)
c2 = BatchNormalization()(c2)
c2 = Activation('relu')(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), padding='same', use_bias=False) (p2)
c3 = BatchNormalization()(c3)
c3 = Activation('relu')(c3)
c3 = Conv2D(32, (3, 3), padding='same', use_bias=False) (c3)
c3 = BatchNormalization()(c3)
c3 = Activation('relu')(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), padding='same', use_bias=False) (p3)
c4 = BatchNormalization()(c4)
c4 = Activation('relu')(c4)
c4 = Conv2D(64, (3, 3), padding='same', use_bias=False) (c4)
c4 = BatchNormalization()(c4)
c4 = Activation('relu')(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), padding='same', use_bias=False) (p4)
c5 = BatchNormalization()(c5)
c5 = Activation('relu')(c5)
c5 = Conv2D(128, (3, 3), padding='same', use_bias=False) (c5)
c5 = BatchNormalization()(c5)
c5 = Activation('relu')(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), padding='same', use_bias=False) (u6)
c6 = BatchNormalization()(c6)
c6 = Activation('relu')(c6)
c6 = Conv2D(64, (3, 3), padding='same', use_bias=False) (c6)
c6 = BatchNormalization()(c6)
c6 = Activation('relu')(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), padding='same',use_bias=False) (u7)
c7 = BatchNormalization()(c7)
c7 = Activation('relu')(c7)
c7 = Conv2D(32, (3, 3), padding='same', use_bias=False) (c7)
c7 = BatchNormalization()(c7)
c7 = Activation('relu')(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), padding='same', use_bias=False) (u8)
c8 = BatchNormalization()(c8)
c8 = Activation('relu')(c8)
c8 = Conv2D(16, (3, 3), padding='same', use_bias=False) (c8)
c8 = BatchNormalization()(c8)
c8 = Activation('relu')(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1])
c9 = Conv2D(8, (3, 3), padding='same', use_bias=False) (u9)
c9 = BatchNormalization()(c9)
c9 = Activation('relu')(c9)
c9 = Conv2D(8, (3, 3), padding='same', use_bias=False) (c9)
c9 = BatchNormalization()(c9)
c9 = Activation('relu')(c9)
    
output_img = Conv2D(filters =1, kernel_size=3, padding='same') (c9)
subtracted = keras.layers.Subtract()([inputs, output_img])

unet_model = Model(inputs=[inputs], outputs=[subtracted])
unet_model.compile(optimizer='adam', loss='mse', metrics=[tf_psnr])

```

```{pytohn}
unet_model.summary()
```

```{python}
Model: "model_3"
__________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==========================================================================================
input_3 (InputLayer)            (None, None, None, 1 0                                            
__________________________________________________________________________________________
conv2d_39 (Conv2D)              (None, None, None, 8 80          input_3[0][0]                    
__________________________________________________________________________________________
activation_37 (Activation)      (None, None, None, 8 0           conv2d_39[0][0]                  
__________________________________________________________________________________________
conv2d_40 (Conv2D)              (None, None, None, 8 576         activation_37[0][0]              
__________________________________________________________________________________________
batch_normalization_35 (BatchNo (None, None, None, 8 32          conv2d_40[0][0]                  
__________________________________________________________________________________________
activation_38 (Activation)      (None, None, None, 8 0           batch_normalization_35[0][0]     
__________________________________________________________________________________________
max_pooling2d_9 (MaxPooling2D)  (None, None, None, 8 0           activation_38[0][0]              
__________________________________________________________________________________________
conv2d_41 (Conv2D)              (None, None, None, 1 1152        max_pooling2d_9[0][0]            
__________________________________________________________________________________________
batch_normalization_36 (BatchNo (None, None, None, 1 64          conv2d_41[0][0]                  
__________________________________________________________________________________________
activation_39 (Activation)      (None, None, None, 1 0           batch_normalization_36[0][0]     
__________________________________________________________________________________________
conv2d_42 (Conv2D)              (None, None, None, 1 2304        activation_39[0][0]              
__________________________________________________________________________________________
batch_normalization_37 (BatchNo (None, None, None, 1 64          conv2d_42[0][0]                  
__________________________________________________________________________________________
activation_40 (Activation)      (None, None, None, 1 0           batch_normalization_37[0][0]     
__________________________________________________________________________________________
max_pooling2d_10 (MaxPooling2D) (None, None, None, 1 0           activation_40[0][0]              
__________________________________________________________________________________________
conv2d_43 (Conv2D)              (None, None, None, 3 4608        max_pooling2d_10[0][0]           
__________________________________________________________________________________________
batch_normalization_38 (BatchNo (None, None, None, 3 128         conv2d_43[0][0]                  
__________________________________________________________________________________________
activation_41 (Activation)      (None, None, None, 3 0           batch_normalization_38[0][0]     
__________________________________________________________________________________________
conv2d_44 (Conv2D)              (None, None, None, 3 9216        activation_41[0][0]              
__________________________________________________________________________________________
batch_normalization_39 (BatchNo (None, None, None, 3 128         conv2d_44[0][0]                  
__________________________________________________________________________________________
activation_42 (Activation)      (None, None, None, 3 0           batch_normalization_39[0][0]     
__________________________________________________________________________________________
max_pooling2d_11 (MaxPooling2D) (None, None, None, 3 0           activation_42[0][0]              
__________________________________________________________________________________________
conv2d_45 (Conv2D)              (None, None, None, 6 18432       max_pooling2d_11[0][0]           
__________________________________________________________________________________________
batch_normalization_40 (BatchNo (None, None, None, 6 256         conv2d_45[0][0]                  
__________________________________________________________________________________________
activation_43 (Activation)      (None, None, None, 6 0           batch_normalization_40[0][0]     
__________________________________________________________________________________________
conv2d_46 (Conv2D)              (None, None, None, 6 36864       activation_43[0][0]              
__________________________________________________________________________________________
batch_normalization_41 (BatchNo (None, None, None, 6 256         conv2d_46[0][0]                  
__________________________________________________________________________________________
activation_44 (Activation)      (None, None, None, 6 0           batch_normalization_41[0][0]     
__________________________________________________________________________________________
max_pooling2d_12 (MaxPooling2D) (None, None, None, 6 0           activation_44[0][0]              
__________________________________________________________________________________________
conv2d_47 (Conv2D)              (None, None, None, 1 73728       max_pooling2d_12[0][0]           
__________________________________________________________________________________________
batch_normalization_42 (BatchNo (None, None, None, 1 512         conv2d_47[0][0]                  
__________________________________________________________________________________________
activation_45 (Activation)      (None, None, None, 1 0           batch_normalization_42[0][0]     
__________________________________________________________________________________________
conv2d_48 (Conv2D)              (None, None, None, 1 147456      activation_45[0][0]              
__________________________________________________________________________________________
batch_normalization_43 (BatchNo (None, None, None, 1 512         conv2d_48[0][0]                  
__________________________________________________________________________________________
activation_46 (Activation)      (None, None, None, 1 0           batch_normalization_43[0][0]     
__________________________________________________________________________________________
conv2d_transpose_9 (Conv2DTrans (None, None, None, 6 32832       activation_46[0][0]              
__________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, None, None, 1 0           conv2d_transpose_9[0][0]         
                                                                 activation_44[0][0]              
__________________________________________________________________________________________
conv2d_49 (Conv2D)              (None, None, None, 6 73728       concatenate_9[0][0]              
__________________________________________________________________________________________
batch_normalization_44 (BatchNo (None, None, None, 6 256         conv2d_49[0][0]                  
__________________________________________________________________________________________
activation_47 (Activation)      (None, None, None, 6 0           batch_normalization_44[0][0]     
__________________________________________________________________________________________
conv2d_50 (Conv2D)              (None, None, None, 6 36864       activation_47[0][0]              
__________________________________________________________________________________________
batch_normalization_45 (BatchNo (None, None, None, 6 256         conv2d_50[0][0]                  
__________________________________________________________________________________________
activation_48 (Activation)      (None, None, None, 6 0           batch_normalization_45[0][0]     
__________________________________________________________________________________________
conv2d_transpose_10 (Conv2DTran (None, None, None, 3 8224        activation_48[0][0]              
__________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, None, None, 6 0           conv2d_transpose_10[0][0]        
                                                                 activation_42[0][0]              
__________________________________________________________________________________________
conv2d_51 (Conv2D)              (None, None, None, 3 18432       concatenate_10[0][0]             
__________________________________________________________________________________________
batch_normalization_46 (BatchNo (None, None, None, 3 128         conv2d_51[0][0]                  
__________________________________________________________________________________________
activation_49 (Activation)      (None, None, None, 3 0           batch_normalization_46[0][0]     
__________________________________________________________________________________________
conv2d_52 (Conv2D)              (None, None, None, 3 9216        activation_49[0][0]              
__________________________________________________________________________________________
batch_normalization_47 (BatchNo (None, None, None, 3 128         conv2d_52[0][0]                  
__________________________________________________________________________________________
activation_50 (Activation)      (None, None, None, 3 0           batch_normalization_47[0][0]     
__________________________________________________________________________________________
conv2d_transpose_11 (Conv2DTran (None, None, None, 1 2064        activation_50[0][0]              
__________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, None, None, 3 0           conv2d_transpose_11[0][0]        
                                                                 activation_40[0][0]              
__________________________________________________________________________________________
conv2d_53 (Conv2D)              (None, None, None, 1 4608        concatenate_11[0][0]             
__________________________________________________________________________________________
batch_normalization_48 (BatchNo (None, None, None, 1 64          conv2d_53[0][0]                  
__________________________________________________________________________________________
activation_51 (Activation)      (None, None, None, 1 0           batch_normalization_48[0][0]     
__________________________________________________________________________________________
conv2d_54 (Conv2D)              (None, None, None, 1 2304        activation_51[0][0]              
__________________________________________________________________________________________
batch_normalization_49 (BatchNo (None, None, None, 1 64          conv2d_54[0][0]                  
__________________________________________________________________________________________
activation_52 (Activation)      (None, None, None, 1 0           batch_normalization_49[0][0]     
__________________________________________________________________________________________
conv2d_transpose_12 (Conv2DTran (None, None, None, 8 520         activation_52[0][0]              
__________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, None, None, 1 0           conv2d_transpose_12[0][0]        
                                                                 activation_38[0][0]              
__________________________________________________________________________________________
conv2d_55 (Conv2D)              (None, None, None, 8 1152        concatenate_12[0][0]             
__________________________________________________________________________________________
batch_normalization_50 (BatchNo (None, None, None, 8 32          conv2d_55[0][0]                  
__________________________________________________________________________________________
activation_53 (Activation)      (None, None, None, 8 0           batch_normalization_50[0][0]     
__________________________________________________________________________________________
conv2d_56 (Conv2D)              (None, None, None, 8 576         activation_53[0][0]              
__________________________________________________________________________________________
batch_normalization_51 (BatchNo (None, None, None, 8 32          conv2d_56[0][0]                  
__________________________________________________________________________________________
activation_54 (Activation)      (None, None, None, 8 0           batch_normalization_51[0][0]     
__________________________________________________________________________________________
conv2d_57 (Conv2D)              (None, None, None, 1 73          activation_54[0][0]              
__________________________________________________________________________________________
subtract_3 (Subtract)           (None, None, None, 1 0           input_3[0][0]                    
                                                                 conv2d_57[0][0]                  
==========================================================================================
Total params: 487,921
Trainable params: 486,465
Non-trainable params: 1,456
__________________________________________________________________________________________________
```



Dncnn

```{python}
#DnCNN
inputs = Input((None, None,1))

c1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (inputs)
c2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c1)

c3 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c2)
c4 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c3)

c5 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c4)
c6 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c5)

c7 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c6)
c8 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c7)


c9 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c8)
c10 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c9)


c11 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c10)
c12 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c11)

c13 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c12)
c14 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c13)


c15 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c14)
c16 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same') (c15)

output_img = Conv2D(filters =1, kernel_size=3, padding='same') (c16)
subtracted = keras.layers.Subtract()([inputs, output_img])


dncnn_model = Model(inputs=[inputs], outputs=[subtracted])
dncnn_model.compile(optimizer='adam', loss='mse', metrics=[tf_psnr])

```



```{python}
Model: "model_2"
__________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==========================================================================================
input_2 (InputLayer)            (None, None, None, 1 0                                            
__________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, None, None, 6 640         input_2[0][0]                    
__________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, None, None, 6 36928       conv2d_18[0][0]                  
__________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, None, None, 6 36928       conv2d_19[0][0]                  
__________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, None, None, 6 36928       conv2d_20[0][0]                  
__________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, None, None, 6 36928       conv2d_21[0][0]                  
__________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, None, None, 6 36928       conv2d_22[0][0]                  
__________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, None, None, 6 36928       conv2d_23[0][0]                  
__________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, None, None, 6 36928       conv2d_24[0][0]                  
__________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, None, None, 6 36928       conv2d_25[0][0]                  
__________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, None, None, 6 36928       conv2d_26[0][0]                  
__________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, None, None, 6 36928       conv2d_27[0][0]                  
__________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, None, None, 6 36928       conv2d_28[0][0]                  
__________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, None, None, 6 36928       conv2d_29[0][0]                  
__________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, None, None, 6 36928       conv2d_30[0][0]                  
__________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, None, None, 6 36928       conv2d_31[0][0]                  
__________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, None, None, 6 36928       conv2d_32[0][0]                  
__________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, None, None, 1 577         conv2d_33[0][0]                  
__________________________________________________________________________________________
subtract_2 (Subtract)           (None, None, None, 1 0           input_2[0][0]                    
                                                                 conv2d_34[0][0]                  
==========================================================================================
Total params: 555,137
Trainable params: 555,137
Non-trainable params: 0
__________________________________________________________________________________________
```





```{python}
inputs = Input((None, None,1))
c1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu') (inputs)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same' , use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

c1 = Conv2D(filters=64, kernel_size=3, padding='same', use_bias = False) (c1)
c1 = BatchNormalization()(c1)
c1 = Activation('relu')(c1)

output_img = Conv2D(filters=1, kernel_size=3, padding='same') (c1)
subtracted = keras.layers.Subtract()([inputs, output_img])
dncnn_model = Model(inputs=[inputs], outputs=[subtracted])
dncnn_model.compile(optimizer='adam', loss='mse', metrics=[tf_psnr])

```

```{python}
Model: "model_1"
__________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==========================================================================================
input_2 (InputLayer)            (None, None, None, 1 0                                            
__________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, None, None, 6 640         input_2[0][0]                    
__________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, None, None, 6 36864       conv2d_18[0][0]                  
__________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, None, None, 6 256         conv2d_19[0][0]                  
__________________________________________________________________________________________
activation_16 (Activation)      (None, None, None, 6 0           batch_normalization_16[0][0]     
__________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, None, None, 6 36864       activation_16[0][0]              
__________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, None, None, 6 256         conv2d_20[0][0]                  
__________________________________________________________________________________________
activation_17 (Activation)      (None, None, None, 6 0           batch_normalization_17[0][0]     
__________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, None, None, 6 36864       activation_17[0][0]              
__________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, None, None, 6 256         conv2d_21[0][0]                  
__________________________________________________________________________________________
activation_18 (Activation)      (None, None, None, 6 0           batch_normalization_18[0][0]     
__________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, None, None, 6 36864       activation_18[0][0]              
__________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, None, None, 6 256         conv2d_22[0][0]                  
__________________________________________________________________________________________
activation_19 (Activation)      (None, None, None, 6 0           batch_normalization_19[0][0]     
__________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, None, None, 6 36864       activation_19[0][0]              
__________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, None, None, 6 256         conv2d_23[0][0]                  
__________________________________________________________________________________________
activation_20 (Activation)      (None, None, None, 6 0           batch_normalization_20[0][0]     
__________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, None, None, 6 36864       activation_20[0][0]              
__________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, None, None, 6 256         conv2d_24[0][0]                  
__________________________________________________________________________________________
activation_21 (Activation)      (None, None, None, 6 0           batch_normalization_21[0][0]     
__________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, None, None, 6 36864       activation_21[0][0]              
__________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, None, None, 6 256         conv2d_25[0][0]                  
__________________________________________________________________________________________
activation_22 (Activation)      (None, None, None, 6 0           batch_normalization_22[0][0]     
__________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, None, None, 6 36864       activation_22[0][0]              
__________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, None, None, 6 256         conv2d_26[0][0]                  
__________________________________________________________________________________________
activation_23 (Activation)      (None, None, None, 6 0           batch_normalization_23[0][0]     
__________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, None, None, 6 36864       activation_23[0][0]              
__________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, None, None, 6 256         conv2d_27[0][0]                  
__________________________________________________________________________________________
activation_24 (Activation)      (None, None, None, 6 0           batch_normalization_24[0][0]     
__________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, None, None, 6 36864       activation_24[0][0]              
__________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, None, None, 6 256         conv2d_28[0][0]                  
__________________________________________________________________________________________
activation_25 (Activation)      (None, None, None, 6 0           batch_normalization_25[0][0]     
__________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, None, None, 6 36864       activation_25[0][0]              
__________________________________________________________________________________________
batch_normalization_26 (BatchNo (None, None, None, 6 256         conv2d_29[0][0]                  
__________________________________________________________________________________________
activation_26 (Activation)      (None, None, None, 6 0           batch_normalization_26[0][0]     
__________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, None, None, 6 36864       activation_26[0][0]              
__________________________________________________________________________________________
batch_normalization_27 (BatchNo (None, None, None, 6 256         conv2d_30[0][0]                  
__________________________________________________________________________________________
activation_27 (Activation)      (None, None, None, 6 0           batch_normalization_27[0][0]     
__________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, None, None, 6 36864       activation_27[0][0]              
__________________________________________________________________________________________
batch_normalization_28 (BatchNo (None, None, None, 6 256         conv2d_31[0][0]                  
__________________________________________________________________________________________
activation_28 (Activation)      (None, None, None, 6 0           batch_normalization_28[0][0]     
__________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, None, None, 6 36864       activation_28[0][0]              
__________________________________________________________________________________________
batch_normalization_29 (BatchNo (None, None, None, 6 256         conv2d_32[0][0]                  
__________________________________________________________________________________________
activation_29 (Activation)      (None, None, None, 6 0           batch_normalization_29[0][0]     
__________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, None, None, 6 36864       activation_29[0][0]              
__________________________________________________________________________________________
batch_normalization_30 (BatchNo (None, None, None, 6 256         conv2d_33[0][0]                  
__________________________________________________________________________________________
activation_30 (Activation)      (None, None, None, 6 0           batch_normalization_30[0][0]     
__________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, None, None, 1 577         activation_30[0][0]              
__________________________________________________________________________________________
subtract_2 (Subtract)           (None, None, None, 1 0           input_2[0][0]                    
                                                                 conv2d_34[0][0]                  
==================================================================================================
Total params: 558,017
Trainable params: 556,097
Non-trainable params: 1,920
______________________________
```


# DCAE


# Deep Convolutional Autoencoder For Estimation Of Nonstationary Noise In Images, [link to the paper](https://ieeexplore.ieee.org/abstract/document/8946273?casa_token=AqijcHGSb7QAAAAA:6PdlPq_RGyVlB1Y3ff2oYpSeozwmTyyxw8i1W6MDJ2TqhIgnrPJs6d-H8NAzE4eIaO5iKjvA-H5jtBQ) 
 ##  Abstract
A precise estimation of noise parameters is very important in many image processing applications, such as denoising, deblurring, compression, etc. This problem is well studied for the case of stationary noise in images, and much less studied for the case of nonstationary noise. In this paper, we develop an efficient method of nonstationary noise variance estimation in image regions, based on specially designed deep convolutional autoencoder (DCAE) with a small dimensionality reduction. Training of the proposed DCAE is carried out for a large set of image blocks, including fragments of noise free textures, faces and texts. 

+ **Flow-chart of the proposed BENNV on the base of autoencoder**
![DCAE_scheme](https://user-images.githubusercontent.com/31028574/188620954-8f9c3e1a-aa9b-4ed9-b0b4-be933228282c.PNG)

### Training ###
Run ``` DCAE_train.py ``` with train dataset
### Tetsing ###
Run ``` DCAE_test.py ``` with any test image. one test image t256.png is provided.

## Train data preparation ##
The training set for DCAE is created in the following way: <br />
1) 200 images are taken from the noise-free image database TAMPERE17 [1] (1000 test blocks of size 16x16 are extracted from each test image). <br />
2) 203 images are taken from the faces image database IMDB-WIKI [2] (1000 test blocks of size 16x16 are extracted from each test image). <br />
3) 489 images are taken from the image database NEOCR [3] (1300 test blocks of size 16x16 are extracted from each test image). <br />
In this way, 858700 block are included in the training set, which are named A1.mat A2.mat and A3.mat in the train script.

For each image four scales are created by downsamplimg the images: 1:1, 1:2, 1:4 and 1:8. 60% of blocks are extracted from the scale 1:1, 25% are extracted from the scale 1:2, 10% are extracted from the scale 1:4, 5% are extracted from the scale 1:8. Blocks are selected in a random manner, but for blocks with a larger local variance the probability to be selected is higher.

##  References ##
[1] Ponomarenko, M., Gapon, N., Voronin, V., & Egiazarian, K. (2018). Blind estimation of white Gaussian noise variance in highly textured images. Electronic Imaging, 2018(13), 1-5. <br />
[2] Rothe, R., Timofte, R. & Gool, L.V. Deep expectation of real and apparent age from a single image without facial landmarks. International Journal of Computer Vision (IJCV), 2016. <br />
[3] R. Nagy, A. Dicker and K. Meyer̺Wegener, "NEOCR: A Configurable Dataset for Natural Image Text Recognition". In CBDAR Workshop 2011 at ICDAR 2011. pp. 53̺58, September 2011. <br />

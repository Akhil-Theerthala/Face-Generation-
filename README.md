## About the project

This project aims to generate faces based on the faces of celebrities. 
The project is broken down into a series of tasks from loading in data to defining and training adversarial networks.
At the end of the project, the results of the trained Generator are presented to see how it performs. The generated samples are fairly realistic faces with noticable distortions.

---

**Dataset:** (Large-scale CelebFaces Attributes (CelebA) Dataset)[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html]

---
## Model Architechture:

### Discriminator: 
```
Discriminator(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
  (conv2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (fc): Linear(in_features=4096, out_features=1, bias=True)
)
```
### Generator:
```
Generator(
  (fc): Linear(in_features=100, out_features=4096, bias=True)
  (deconv1): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (deconv2): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (last): Sequential(
    (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  )
)
```
---

## Obtained Faces

![image](https://user-images.githubusercontent.com/85566221/186476197-8fb77b92-2190-4627-a436-37e6bd6bba35.png)

---

## Observations & Steps for improvement:
* Increasing the number of epochs from 10 to 15 showed minimal effects
* Increasing the number of epochs from 15 to 25 showed the over-fitting.
* One of the common methods that can improve the performance of the data is to increase the available data.
* Adding a dropout before the fully connected layers would also help improve the model.
* In my approach, label smoothing was done when calculating real loss, to get better results.
* According to the article from Machine Learning Mastery, which can be accessed here, adding an Average pooling along with the stride, might help in improving the discriminator performance.


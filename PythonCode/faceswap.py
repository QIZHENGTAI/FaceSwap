import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import torch
import subprocess
#from realesrgan import RealESRGANs
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
print(f"Using device: {device}")

app = FaceAnalysis(name='buffalo_l')  # Use smaller model for speed
app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))  # Ensure GPU usage

swapper = insightface.model_zoo.get_model(
    'PretrainModel/inswapper_128.onnx',
    download=False,
    providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
)



def swap_n_show(img1_fn, img2_fn, app, swapper, plot_before=True, plot_after=True):
    """
    Uses face swapper to swap faces in two different images.

    plot_before: if True shows the images before the swap
    plot_after: if True shows the images after the swap

    returns images with swapped faces.

    Assumes one face per image.
    """
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)

    # if plot_before:
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     axs[0].imshow(img1[:,:,::-1])
    #     axs[0].axis('off')
    #     axs[1].imshow(img2[:,:,::-1])
    #     axs[1].axis('off')
    #     plt.show()

    # Do the swap
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]

    img1_ = img1.copy()
    img2_ = img2.copy()

    # img1_ = cv2.medianBlur(img1_, 5)
    # img2_ = cv2.medianBlur(img2_, 5)


    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img2_ = swapper.get(img2_, face2, face1, paste_back=True)
        # img1_ = cv2.medianBlur(img1_, 3)
        # img2_ = cv2.medianBlur(img2_, 3)

        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # for x in range(0):
        #     img1_ = cv2.filter2D(img1_, -1, kernel)
        #     img2_ = cv2.filter2D(img2_, -1, kernel)
        #     img1_ = cv2.cvtColor(img1_, cv2.COLOR_BGR2YUV)
        #     img1_[:, :, 0] = cv2.equalizeHist(img1_[:, :, 0])
        #     img1_ = cv2.cvtColor(img1_, cv2.COLOR_YUV2BGR)
        #     img2_ = cv2.cvtColor(img2_, cv2.COLOR_BGR2YUV)
        #     img2_[:, :, 0] = cv2.equalizeHist(img2_[:, :, 0])
        #     img2_ = cv2.cvtColor(img2_, cv2.COLOR_YUV2BGR)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1_[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2_[:,:,::-1])
        axs[1].axis('off')
        plt.show()


    return img1_, img2_



img1_fn, img2_fn = "Image/ironman.png", "Image/trump.png"
swapped_img1, swapped_img2 = swap_n_show(img1_fn, img2_fn, app, swapper)

    # Convert numpy arrays to RGB format for saving as images
swapped_img1_rgb = cv2.cvtColor(swapped_img1, cv2.COLOR_BGR2RGB)
swapped_img2_rgb = cv2.cvtColor(swapped_img2, cv2.COLOR_BGR2RGB)

    # Save the swapped images
Image.fromarray(np.uint8(swapped_img1_rgb)).save("swapped_image1.jpg")
Image.fromarray(np.uint8(swapped_img2_rgb)).save("swapped_image2.jpg")

    # Download the swapped images
#files.download("swapped_image1.jpg")
#files.download("swapped_image2.jpg")


import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    # input image, output location, and new maximum matrix rank of color channels
    inputPath = "garfield_upside_down244x300.png"
    compressionRank = 10
    outputPath = inputPath + f"_rank{compressionRank}.png"

    # create RGB matrices from image
    inputImage = Image.open(inputPath)
    inputMatrix = np.array(inputImage)
    imageR = inputMatrix[:, :, 0]
    imageG = inputMatrix[:, :, 1]
    imageB = inputMatrix[:, :, 2]
    
    # perform singular value decomposition
    uR, sR, vTR = np.linalg.svd(imageR, full_matrices=False)
    uG, sG, vTG = np.linalg.svd(imageG, full_matrices=False)
    uB, sB, vTB = np.linalg.svd(imageB, full_matrices=False)

    # compress each color channel to chosen rank, *element-wise multiplication and combine to image array
    rankR = np.linalg.matrix_rank(imageR)
    rankG = np.linalg.matrix_rank(imageG)
    rankB = np.linalg.matrix_rank(imageB)
    if rankR > compressionRank: compR = uR[:, 0:compressionRank] * sR[0:compressionRank] @ vTR[0:compressionRank, :]
    if rankG > compressionRank: compG = uG[:, 0:compressionRank] * sG[0:compressionRank] @ vTG[0:compressionRank, :]
    if rankB > compressionRank: compB = uB[:, 0:compressionRank] * sB[0:compressionRank] @ vTB[0:compressionRank, :]
    compImage = np.dstack((compR, compG, compB))
    compImage = np.clip(compImage, 0, 255)

    # save compressed image
    Image.fromarray(compImage.astype(np.uint8)).save(outputPath)
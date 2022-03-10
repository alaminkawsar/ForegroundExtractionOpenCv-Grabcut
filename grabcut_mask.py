# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from matplotlib import pyplot as plt

image = cv2.imread("input.jpg")
mask = cv2.imread("draw", cv2.IMREAD_GRAYSCALE)
# apply a bitwise mask to show what the rough, approximate mask would
# give us
roughOutput = cv2.bitwise_and(image, image, mask=mask)
# show the rough, approximated output

# any mask values greater than zero should be set to probable
# foreground
mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD

fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")
# apply GrabCut using the the mask segmentation method
start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
	fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)
# loop over the possible GrabCut mask values
for (name, value) in values:
	# construct a mask that for the current value
	print("[INFO] showing mask for '{}'".format(name))
	valueMask = (mask == value).astype("uint8") * 255
	# display the mask so we can visualize it
plt.imshow(valueMask)
plt.show()
 

plt.imshow(roughOutput)
plt.show()

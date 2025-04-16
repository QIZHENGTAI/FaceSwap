import os
import cv2
import numpy as np
from gfpgan import GFPGANer




# restorer = GFPGANer(model_path='PretrainModel/GFPGANv1.3.pth')
# img = cv2.imread('swapped_image1.jpg', cv2.IMREAD_COLOR)

# cropped_faces, restored_img, restored_faces = restorer.enhance(
#     img, has_aligned=False, only_center_face=True
# )
# restored_img = restored_img[0]
# # Now, convert it to a NumPy array (if it's still a PIL Image)
# if isinstance(restored_img, np.ndarray):
#     img_to_save = restored_img
# else:
#     img_to_save = np.array(restored_img)

# cv2.imwrite('restored_face2.jpg', restored_img)



import cv2
import numpy as np
from gfpgan import GFPGANer

def ImageRestore(image_path):
    # Initialize restorer
    restorer = GFPGANer(model_path='PretrainModel/GFPGANv1.3.pth')
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print("Failed to load the image.")
    else:
        try:
            # Enhance face
            cropped_faces, restored_img_list, restored_faces = restorer.enhance(
                img, has_aligned=False, only_center_face=True
            )

            # If output is a list of images, get the first one
            if isinstance(restored_img_list, list):
                restored_img = restored_img_list[0]
            else:
                restored_img = restored_img_list

            # Convert to numpy if needed
            if not isinstance(restored_img, np.ndarray):
                restored_img = np.array(restored_img)

            # Save output
            cv2.imwrite('restored_face2.jpg', restored_img)
            print("Face restored and saved as 'restored_face2.jpg'.")

        except Exception as e:
            print(f"Restoration failed: {e}")

    return restored_img





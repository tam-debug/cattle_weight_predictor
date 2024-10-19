from pathlib import Path

from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from weight_model.mask_mappings import load_dataset

def view_data_aug(depth_masks_path: Path):
    # Example: Load your image as a numpy array
    # Assuming the image is in HxWxC format and with values in [0, 255]
    depth_masks, weights = load_dataset([depth_masks_path])
    depth_masks = np.expand_dims(depth_masks, axis=1)
    depth_masks = np.repeat(depth_masks, 3, axis=1)
    image_np = depth_masks[0]
    image_np = image_np.astype(np.uint8)
    # image_np = image_np[0]
    image_np = np.transpose(image_np, (1, 2, 3, 0))
    image_np = image_np[0]

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_np)

    # Define your augmentations using torchvision.transforms
    transformations = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0)),
        v2.RandomAffine(20),
        v2.RandomPerspective(),
    ]
    # Number of augmented versions to visualize
    num_augmentations = 5

    # Set up the plot
    plt.figure(figsize=(15, 2))

    # Plot the original image
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(image_np)
    plt.title("Original Image", fontsize=10)
    plt.axis('off')

    titles = [
        "Horizontal Flip",
        "Rotation",
        "Resized Crop",
        "Affine",
        "Perspective"
    ]
    augmentations = transforms.Compose(transformations)

    # Apply and plot augmentations
    for i in range(num_augmentations):
        # Apply the augmentation
        # augmentations = transforms.Compose([transformations[i]])
        augmented_image = augmentations(image)

        # Convert the augmented image to numpy array
        augmented_image_np = np.array(augmented_image)

        # Plot the augmented image
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented_image_np)
        # plt.title(titles[i])
        plt.title(f"Augmented image {i+1}", fontsize=10)
        plt.axis('off')

    plt.show()

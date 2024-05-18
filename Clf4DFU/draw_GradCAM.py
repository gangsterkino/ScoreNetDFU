import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

import matplotlib.colors as mcolors
import cv2
def main():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(2048, 4)
    model_weights_path = '/Users/tanxinyu/model/model3.pt'
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    target_layers = [model.layer4]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load image
    img_path = "test/sample1.png"
    assert os.path.exists(img_path), "File '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 3

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    # Remove axis ticks and labels
    plt.axis('off')
    # Create a colorbar to represent relevance
    # Define a custom colormap from purple/blue to orange

    # Define a custom colormap from blue-purple to red-orange
    cmap_colors = [ (0.6, 0, 1), (0, 0, 1),(0, 0.5, 1), (0, 1, .8), (0.8, 1, 0), (1, 0.5, 0),
                   (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('Custom', cmap_colors, N=100)

    # Apply the colormap to the heatmap
    heatmap = cmap(grayscale_cam)

    # ... (Rest of your code)

    # Create a colorbar to represent relevance
    cbar = plt.colorbar(ScalarMappable(cmap=cmap), orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Relevance', rotation=270, labelpad=15)
    cbar.ax.set_yticklabels([])  # Remove colorbar ticks
    # Define class labels (update with your own class labels)
    class_labels = ['gangrene', 'infection', 'normal', 'ulcer']
    # Add a title to the plot
    plt.title(f'{class_labels[target_category]}\'s Grad-CAM Map', fontsize=14)
    plt.show()

if __name__ == '__main__':
    main()

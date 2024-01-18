import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor

class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al. 
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''
    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())
        
    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)                 # Module.to() is in-place method 
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()
        
        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)
        
        # backward
        model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()
        
        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0) # ReLU
        cam = cam / cam.max()
        return cam
    
    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]
        
        cam = cv2.resize(cam, (h,w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255*cam).astype(np.uint8), cv2.COLORMAP_JET) # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        image = image / image.max()
        heatmap = heatmap / heatmap.max()
        
        result = 0.4*heatmap + 0.6*image
        result = result / result.max()
        
        plt.figure()
        plt.imshow((result*255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
        	ToTensor(),
        	Normalize(mean=mean, std=std)
            ])
        return preprocessing(img.copy()).unsqueeze(0) 


if __name__ == '__main__':
    # image = cv2.imread('both.png') # (224,224,3)
    image = np.random.rand(255,255,3).astype(np.float32)
    input_tensor = GradCAM.preprocess_image(image)
    model = models.resnet18(pretrained=True)
    grad_cam = GradCAM(model, model.layer4[-1], 224)
    cam = grad_cam.calculate_cam(input_tensor)
    GradCAM.show_cam_on_image(image, cam)

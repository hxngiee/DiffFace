import torch 
import torch.nn as nn 
from models.gaze_estimation.models.eyenet import EyeNet
from torchvision import transforms

class Gaze_estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.checkpoint = torch.load('checkpoints/GazeEstimator.pt', map_location=self.device)
        self.nstack = self.checkpoint['nstack']
        self.nfeatures = self.checkpoint['nfeatures']
        self.nlandmarks = self.checkpoint['nlandmarks']
        self.eyenet = EyeNet(nstack=self.nstack, nfeatures=self.nfeatures, nlandmarks=self.nlandmarks).to(self.device)
        self.eyenet.load_state_dict(self.checkpoint['model_state_dict'])
        self.t = transforms.Resize((96, 160))
    
    def forward(self, image):
        heatmaps_pred, landmarks_pred, gaze_pred = self.eyenet.forward(self.t(image))
        return gaze_pred 

if __name__ == '__main__':
    model = Gaze_estimator()
import torch
import clip
import torch.nn as nn
# from configs.paths_config import MODEL_PATHS
from models.insight_face.model_irse import Backbone, MobileFaceNet


class IDLoss(nn.Module):
    def __init__(self, use_mobile_id=False):        
        super(IDLoss, self).__init__()
        MODEL_PATHS = "./pre_trained/model_ir_se50.pth"
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(MODEL_PATHS, weights_only=True))
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        self.facenet = self.facenet.to(x.device)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        # print(losses)
        return torch.mean(losses)




class ClipLoss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32",device=config.device)
        self.preprocess.transforms.pop(2)
        self.preprocess.transforms.pop(2)
        self.config = config
        self.model.eval()

        for param in self.model.parameters():
            param.requiers_grad = False


    
    def forward(self,img1, img2):
        img1 = img1.to(self.config.device)
        img2 = img2.to(self.config.device)
        
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)
        
       
        image_features_1 = self.model.encode_image(img1).float()
        image_features_2 = self.model.encode_image(img2).float()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        
        return torch.mean(1-cos(image_features_1, image_features_2)) 
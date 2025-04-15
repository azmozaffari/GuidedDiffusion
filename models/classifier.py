import torch
import clip
import torch.nn as nn
# from configs.paths_config import MODEL_PATHS
from models.insight_face.model_irse import Backbone, MobileFaceNet
from models.emonet import emonet
from transformers import CLIPModel, CLIPProcessor

class IDLoss(nn.Module):
    def __init__(self, config):        
        super(IDLoss, self).__init__()
        MODEL_PATHS =config.checkpoints.pretrained_classifier_FACEID
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
        if config.training.use_finetuned_classifier == "on":
            model_path = config.checkpoints.pretrained_classifier_clip  # Replace with the actual path  
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            
        
        
        
        self.preprocess.transforms.pop(2)
        self.preprocess.transforms.pop(2)
        self.config = config
        self.model.eval()

        for param in self.model.parameters():
            param.requiers_grad = False


    
    def forward(self,img_source, img_generated, text_source, text_target):
        img_source = img_source.to(self.config.device)
        img_generated = img_generated.to(self.config.device)
        
        img_source = self.preprocess(img_source)
        img_generated = self.preprocess(img_generated)
        
       
        image_features_source = self.model.encode_image(img_source).float()
        image_features_generated = self.model.encode_image(img_generated).float()



        # texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
        
        text_s = [text_source]
        text_t = [text_target]

        text_tokens_source = clip.tokenize(text_s).to(self.config.device)
        text_tokens_target = clip.tokenize(text_t).to(self.config.device)
        
        # Compute the image and text features
        text_features_s = self.model.encode_text(text_tokens_source)
        text_features_t = self.model.encode_text(text_tokens_target)
        
        
        text_features_s = text_features_s.repeat(img_source.size(0),text_features_s.size(0))
        text_features_t = text_features_t.repeat(img_source.size(0),text_features_t.size(0))

        # print(text_features.size(), image_features_1.size())

    # Calculate the similarity
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        
        delta_T = text_features_t - text_features_s
        delta_I = image_features_generated - image_features_source
        
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        
        return torch.mean(1 - (cos(delta_I, delta_T)/(delta_I.norm(dim=1, p=2) * delta_T.norm(dim=1, p=2))))
     
    


class EmoNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        MODEL_PATHS = config.checkpoints.pretrained_classifier_emonet
        self.model = emonet(n_expression=5).to(config.device)
        state_dict = torch.load(MODEL_PATHS, weights_only=True, map_location=config.device)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requiers_grad = False

        print("Emonet model is loaded")
    

    def forward(self, img, emotion):
        
        output, feature =self.model(img)
        emotion_value = 1 - nn.functional.softmax(output["expression"]/2, dim=1)[:,emotion]  # dividing the softmax output by T>1 helps to have more calibrated confidence results and can help the model to learn better adversaries to guid image generation 
        #emotion_classes = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}
        return torch.mean(emotion_value), feature

        
    


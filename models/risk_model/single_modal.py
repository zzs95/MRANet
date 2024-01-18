import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchxrayvision as xrv
import torchvision
from dataset.constants import COVID_REGIONS, REPORT_KEYS
from models.losses_surv import CoxPHLoss
import torchtuples as tt
# from models.feat_text_generator.modules.MLP import MLPVanilla
from models.encoders.image.resnet import ResNet, AttentionPool2d
# from torchvision.models import resnet50, ResNet50_Weights
    
class SingleModalModel(nn.Module):
    """
    Full model consisting of:
        - object detector encoder
        - binary classifier for selecting regions for sentence genneration
        - binary classifier for detecting if a region is abnormal or normal (to encode this information in the region feature vectors)
        - language model decoder
    """

    def __init__(self, local_rank=0):
        super().__init__()
        self.local_rank = local_rank 
        self.num_queries = len(REPORT_KEYS)
        self.risk_embed_dim = 2048
        # self.img_encoder = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='mean', img_size=512)
        
        self.img_encoder = ResNet(name='resnet50', in_channels=1, pretrained=False, pool_method='mean') # 224
        CHECKPOINT = './checkpoints/prior_resnet50.pt'
        checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
        from collections import OrderedDict
        new_state_dict = OrderedDict()        
        for k, v in checkpoint.items():
            new_state_dict['encoder.'+k] = v
        self.img_encoder.load_state_dict(new_state_dict)
        for k,v in self.img_encoder.named_parameters():
            if k.replace('encoder.', '') in checkpoint.keys():
                v.requires_grad = False   

        self.global_attention_pooler = AttentionPool2d(in_features=2048, feat_size=7, embed_dim=2048, num_heads=8) 
        
        # resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # self.img_encoder.out_channels = 2048
        
        # self.img_encoder = xrv.models.ResNet(weights="resnet50-res512-all")
        
        # self.clin_risk_feat_encoder = MLPVanilla(in_features=16, num_nodes=[self.risk_embed_dim], out_features=None,
        #                                             batch_norm=True, dropout=0.1, output_bias=False, output_activation=None)  
        
        
        # gatortron_dim = 2560 # medium
        # self.text_risk_feat_encoder = nn.Sequential(nn.Flatten(start_dim=1), 
                                            # MLPVanilla(in_features=gatortron_dim * 5, num_nodes=[self.risk_embed_dim, self.risk_embed_dim], out_features=None,
                                            # batch_norm=True, dropout=0.1, output_bias=False, output_activation=None)  )
        
        self.risk_predictor = nn.Sequential(nn.Linear(self.risk_embed_dim*1, 1), torch.nn.Sigmoid())
        self.criterion_surv = CoxPHLoss()
        self.lambda_surv = 0.1
      
    def stage1_step(self, batch, device, stage='stage1', return_score=False):
        batch_size = batch["clin_feat"].size(0)
        # clin_feat = batch["clin_feat"].to(device, non_blocking=True)
        # risk_feat = self.clin_risk_feat_encoder(clin_feat)
        
        global_feat = batch["images"].to(device, non_blocking=True)
        # piror resnet mean pool
        # local_features, risk_feat = self.img_encoder(global_feat) 
        # piror resnet attention pool
        with torch.no_grad():
            local_features, global_x, features = self.img_encoder(global_feat, return_features=True)
        
        # local_features = batch["images"].to(device, non_blocking=True)
        risk_feat = self.global_attention_pooler(local_features)
        
        # xrv
        # global_feat = (global_feat /255) * 2048 - 1024
        # risk_feat = self.img_encoder.features(global_feat) 
        # normal resnet
        # risk_feat = self.img_encoder(global_feat) 
        
        # text_feat = batch["text_feats"].to(device, non_blocking=True)
        # risk_feat = self.text_risk_feat_encoder(text_feat)
        
        risk_score = self.risk_predictor(risk_feat.reshape(batch_size, -1))
        if return_score:
            return risk_score
        risk_score = tt.tuplefy(risk_score.float())
        time_label = batch["time_label"].float().to(device, non_blocking=True)
        event_label = batch["event_label"].long().to(device, non_blocking=True)
        surv_label = (time_label, event_label)
        surv_label = tt.tuplefy(surv_label).to_device(device)
        loss_surv = self.criterion_surv(*risk_score, *surv_label)
        
        loss_dict = {}
        loss_dict['loss'] = 0
        loss_dict['surv_loss'] = loss_surv 
        loss_dict['loss'] += loss_surv * self.lambda_surv
        return loss_dict
    

    def forward(self, run_params, batch, device, ):
        loss_dict = self.stage1_step(batch, device)
        loss = loss_dict['loss']
        return loss, loss_dict

    @torch.no_grad()
    def risk_predict(self, run_params, batch, device, return_feats=False):
        risk_score = self.stage1_step(batch, device, return_score=True)
        return risk_score
        

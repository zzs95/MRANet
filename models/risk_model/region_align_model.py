import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataset.constants import COVID_REGIONS, REPORT_KEYS, REGION_IDS
from models.losses_surv import CoxPHLoss
import torchtuples as tt
from models.MLP import MLPVanilla
from models.encoders.image.resnet import ResNet, AttentionPool2d
# from torchvision.models import resnet50, ResNet50_Weights
# import torchxrayvision as xrv
from models.language_model import LanguageModel

class MLPHead(nn.Module):
    def __init__(self, in_channels, representation_size, out_relu=True):
        super().__init__()
        self.out_relu = out_relu
        self.fc5 = nn.Linear(in_channels, representation_size)
        self.fc6 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        if self.out_relu:
            x = F.relu(x)
        return x   
    
class CrossAlignModel(nn.Module):
    """
    Full model consisting of:
        - object detector encoder
        - binary classifier for selecting regions for sentence genneration
        - binary classifier for detecting if a region is abnormal or normal (to encode this information in the region feature vectors)
        - language model decoder
    """

    def __init__(self, stage='stage_2'):
        super().__init__()
        self.num_queries = len(REPORT_KEYS)
        self.embed_dim = 1024
        self.region_ids = REGION_IDS

        self.representation_size = 1024
        self.global_resolution = 16
        self.img_size = 224
        self.risk_embed_dim = 2048
        self.img_encoder = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='mean')
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
        # MRE
        self.roi_embed_dim = 256
        self.roi_embedder = nn.ModuleList()
        self.roi_resolution_list = [4,2,2,1,1]
        self.roi_channel_list = [64, 256, 512, 1024, 2048]
        for i in range(len(self.roi_channel_list)):
            in_feat_len = self.roi_channel_list[i] * self.roi_resolution_list[i]**2
            self.roi_embedder.append(MLPHead(in_feat_len, self.roi_embed_dim))
        
        self.region_visual_embedder = nn.ModuleList()
        for k, v in COVID_REGIONS.items():
            if k == 'Bones':
                self.bone_global_embedder = nn.Linear(2048, self.roi_embed_dim * len(self.roi_channel_list))
                v = v + [29,]
            self.region_visual_embedder.append(MLPHead(len(v) * self.roi_embed_dim * len(self.roi_channel_list), self.representation_size))
        self.region_visual_embedder.append(MLPHead(2048, self.representation_size)) # impression, global_feat
        
        # SSE
        self.survival_attention_pooler = AttentionPool2d(in_features=2048, feat_size=7, embed_dim=self.risk_embed_dim, num_heads=8)  #  global_feat --> risk_feat
        self.region_risk_embedder = nn.ModuleList()
        for k, v in COVID_REGIONS.items():
            self.region_risk_embedder.append(MLPHead(self.risk_embed_dim, self.representation_size) )
        self.region_risk_embedder.append(MLPHead(self.risk_embed_dim, self.representation_size)) # impression
        
        # text
        self.gatortron_dim = 2560
        self.gatortron_projection = MLPVanilla(in_features=self.gatortron_dim, num_nodes=[self.embed_dim , self.embed_dim], out_features=None,
                                    batch_norm=True, dropout=0.1, output_bias=False, output_activation=None)
        self.step = 0
        
        # img_no_MRE
        # self.global_visual_embedder = nn.ModuleList()
        # for i in range(len(COVID_REGIONS) + 1):
        #     self.global_visual_embedder.append(MLPHead(2048, self.representation_size))
        
        # shared structure
        self.region2sentence_embedder = MLPHead(self.representation_size, self.representation_size)    
        self.criterion_surv = CoxPHLoss()
        self.lambda_surv = 0.1
        self.risk_predictor = nn.Sequential(nn.Linear(self.risk_embed_dim*1, 1), torch.nn.Sigmoid())       
        self.feature_space_transformation_nn = MLPHead(self.embed_dim, self.embed_dim, out_relu=False)   
        self.stage = stage
        if stage=='stage_2' or stage == 'pred':
            self.language_model = LanguageModel()
        if stage=='stage_3' or stage == 'pred':
            self.clin_risk_feat_encoder = MLPVanilla(in_features=16, num_nodes=[self.risk_embed_dim,self.risk_embed_dim], out_features=None,
                                batch_norm=True, dropout=0.1, output_bias=False, output_activation=None)
        
            self.text_feat_projection = MLPHead(self.num_queries*self.embed_dim, self.risk_embed_dim, out_relu=False)
            self.imgtext_fuse_projection = MLPHead(self.risk_embed_dim*2, self.risk_embed_dim, out_relu=False)
            self.clin_fuse_projection =  MLPHead(self.risk_embed_dim*2, self.risk_embed_dim, out_relu=False)

    def forward(self, batch, device):
        if self.stage == 'stage_2':
            loss_dict = self.stage2_step(batch, device)
        elif self.stage == 'stage_3':
            loss_dict = self.stage3_step(batch, device)
        return loss_dict                        
    
    def region_sentence_encode(self, roi_feats, global_feat, risk_feat, risk_guide=True ):
        sentence_feat_embed = []
        for i, [k, v] in enumerate(COVID_REGIONS.items()):
            if k == 'Lines/tubes' or k == 'Others':
                sentence_feat_embed.append(None)
            elif k == 'Lungs' or k == 'Pleura' or k =='Heart and mediastinum' or k == 'Bones': 
                region_features_tmp = torch.index_select(roi_feats, 1, torch.LongTensor(v).to(roi_feats.device)).reshape([roi_feats.shape[0], -1])
                if k == 'Bones':
                    region_features_tmp = torch.concat([region_features_tmp, self.bone_global_embedder(global_feat)], dim=1)
                if risk_guide == False:
                    sentence_feat_embed.append(self.region2sentence_embedder(self.region_visual_embedder[i](region_features_tmp))[:,None])
                else:
                    sentence_feat_embed.append(self.region2sentence_embedder(self.region_visual_embedder[i](region_features_tmp) + self.region_risk_embedder[i](risk_feat))[:,None]) 
                
        # IMPRESSION
        if risk_guide == False:
            sentence_feat_embed.append(self.region2sentence_embedder(self.region_visual_embedder[-1](global_feat))[:,None]) 
        else:
            sentence_feat_embed.append(self.region2sentence_embedder(self.region_visual_embedder[-1](global_feat) + self.region_risk_embedder[-1](risk_feat))[:,None]) 
        sentence_feat_embed = torch.concat(sentence_feat_embed, dim=1)
        return sentence_feat_embed
    
    def img_feat_encode(self, batch, device, mean_pool=False):
        images = batch["images"].to(device, non_blocking=True)
        with torch.no_grad():
            local_features, global_x, features = self.img_encoder.encoder(images, return_features=True)
            survival_features = self.survival_attention_pooler(local_features)
        if mean_pool:
            return survival_features, global_x, features
        else:
            return survival_features, features
    
    def region_encode(self, batch, global_feat, risk_feat, multi_scale_features, device, risk_guide=True):
        # roi_align
        batch_size = len(batch["boxes"])
        batch['boxes'] = [batch['boxes'][i].to(device, non_blocking=True) for i in range(batch_size)]
        multi_scale_embed = []
        for i, feat in enumerate(multi_scale_features):
            scale = feat.shape[-1] / self.img_size
            output_size = self.roi_resolution_list[i]
            multi_scale_embed.append(self.roi_embedder[i](
                torchvision.ops.roi_align(input=feat, boxes=batch['boxes'], output_size=output_size, spatial_scale=scale, aligned=False).squeeze()).reshape(batch_size, 29, 1, -1))
        rois_embed = torch.concat(multi_scale_embed, dim=2).reshape(batch_size, batch['boxes'][0].shape[0], -1)
        image_sentence_embed = self.region_sentence_encode(rois_embed, global_feat, risk_feat, risk_guide)
        return image_sentence_embed
    
    def no_region_encode(self, global_feat, risk_feat, risk_guide=True):
        # no_region
        image_sentence_embed = []
        for i in range(len(self.global_visual_embedder)):
            if risk_guide == False:
                image_sentence_embed.append(self.region2sentence_embedder(self.global_visual_embedder[i](global_feat)[:,None]))
            else:
                image_sentence_embed.append(self.region2sentence_embedder(self.global_visual_embedder[i](global_feat) + self.region_risk_embedder[i](risk_feat))[:,None])
        image_sentence_embed = torch.concat(image_sentence_embed, dim=1)
        return image_sentence_embed
        
    
    def stage2_step(self, batch, device):
        batch_size = batch["clin_feat"].size(0)
        # GPT
        gpt_input_ids = batch["input_ids"].to(device, non_blocking=True)
        gpt_attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.no_grad():
            risk_feat, global_feat, features = self.img_feat_encode(batch, device, mean_pool=True)
            risk_score = self.risk_predictor(risk_feat.reshape(batch_size, -1))
            risk_score = tt.tuplefy(risk_score.float())
            time_label = batch["time_label"].float().to(device, non_blocking=True)
            event_label = batch["event_label"].long().to(device, non_blocking=True)
            surv_label = (time_label, event_label)
            surv_label = tt.tuplefy(surv_label).to_device(device)
            loss_surv = self.criterion_surv(*risk_score, *surv_label)
        
        # img + MRE + SSE + text_align
        if self.step % 2 == 0:
            image_sentence_embed = self.region_encode(batch, global_feat, risk_feat, features, device)
            text_hidden_states = self.feature_space_transformation_nn(image_sentence_embed.view(batch_size * self.num_queries, self.embed_dim)) 
            lm_logits, language_model_image_loss = self.language_model(
                input_ids=gpt_input_ids,
                attention_mask=gpt_attention_mask,
                image_hidden_states=text_hidden_states.view(batch_size * self.num_queries, self.embed_dim),
                return_loss=True )      
            language_model_text_loss = torch.tensor(0.0).cuda()
            self.step += 1
        else:
            gatortron_text_embed = batch["text_feats"].to(device, non_blocking=True)
            text_hidden_states = self.gatortron_projection(gatortron_text_embed.reshape(batch_size*self.num_queries, self.gatortron_dim)).reshape(batch_size, self.num_queries, self.embed_dim)
            lm_logits, language_model_text_loss = self.language_model(
                input_ids=gpt_input_ids,
                attention_mask=gpt_attention_mask,
                image_hidden_states=text_hidden_states.view(batch_size * self.num_queries, self.embed_dim),
                return_loss=True )
            language_model_image_loss = torch.tensor(0.0).cuda()
            self.step += 1
        
        # no text align
        # img + MRE + SSE
        # image_sentence_embed = self.region_encode(batch, global_feat, risk_feat, features, device)
        
        # img + MRE
        # image_sentence_embed = self.region_encode(batch, global_feat, None, features, device, risk_guide=False)

        # img + no_MRE 
        # image_sentence_embed = self.no_region_encode(global_feat, None, risk_guide=False) # img + no_MRE + no_SSE (img_only)
        # image_sentence_embed = self.no_region_encode(global_feat, risk_feat, risk_guide=True) # img + no_MRE + SSE

        # text_hidden_states = self.feature_space_transformation_nn(image_sentence_embed.view(batch_size * self.num_queries, self.embed_dim)) 
        # lm_logits, language_model_image_loss = self.language_model(
        #     input_ids=gpt_input_ids,
        #     attention_mask=gpt_attention_mask,
        #     image_hidden_states=text_hidden_states.view(batch_size * self.num_queries, self.embed_dim),
        #     return_loss=True )      
        # language_model_text_loss = torch.tensor(0.0).cuda() 
        
        loss_dict = {}
        loss_dict['loss'] = 0
        loss_dict['lm_img_loss'] = language_model_image_loss
        loss_dict['loss'] += language_model_image_loss
        loss_dict['lm_text_loss'] = language_model_text_loss
        loss_dict['loss'] += language_model_text_loss
        loss_dict['surv_loss'] = loss_surv 
        return loss_dict
    
    @torch.no_grad()
    def generate_text(
        self,
        batch,
        device,
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
        return_text_feat=False
    ):
        batch_size = batch["clin_feat"].size(0)
        risk_feat, global_feat, features = self.img_feat_encode(batch, device, mean_pool=True)
        
        # img + MRE + SSE
        image_sentence_embed = self.region_encode(batch, global_feat, risk_feat, features, device)
        
        # img + MRE
        # image_sentence_embed = self.region_encode(batch, global_feat, None, features, device, risk_guide=False)

        # img + no_MRE + no_SSE (img_only)
        # image_sentence_embed = self.no_region_encode(global_feat, None, risk_guide=False)
         
        # img + no_MRE + SSE
        # image_sentence_embed = self.no_region_encode(global_feat, risk_feat, risk_guide=True) 
        
        text_hidden_states = self.feature_space_transformation_nn(image_sentence_embed.view(batch_size * self.num_queries, self.embed_dim)) 
        if return_text_feat:
            return text_hidden_states.view(batch_size, self.num_queries, self.embed_dim)
        output_ids = self.language_model.generate(
            text_hidden_states.view(batch_size * self.num_queries, self.embed_dim),
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping,
        )
        return output_ids
    
    @ torch.no_grad()
    def generate_img2text_feats(self,
        batch,
        device,
        ):
        return self.generate_text(batch, device, return_text_feat=True)
        
    def stage3_step(self, batch, device, return_score=False):
        batch_size = batch["clin_feat"].size(0)
        clin_feat = batch["clin_feat"].to(device, non_blocking=True)
        clin_risk_feat = self.clin_risk_feat_encoder(clin_feat)
        
        with torch.no_grad():
            risk_feat, global_feat, features = self.img_feat_encode(batch, device, mean_pool=True)
        
            # img + MRE + SSE (text_align)
            image_sentence_embed = self.region_encode(batch, global_feat, risk_feat, features, device)
            img_risk_feat = risk_feat
                        
            # img + MRE
            # image_sentence_embed = self.region_encode(batch, global_feat, None, features, device, risk_guide=False)
            # img_risk_feat = global_feat
            
            # img + no_MRE + no_SSE (img_only)
            # image_sentence_embed = self.no_region_encode(global_feat, None, risk_guide=False) 
            # img_risk_feat = global_feat
            
            # img + no_MRE + SSE
            # image_sentence_embed = self.no_region_encode(global_feat, risk_feat, risk_guide=True) 
            # img_risk_feat = risk_feat

            text_hidden_states = self.feature_space_transformation_nn(image_sentence_embed.view(batch_size * self.num_queries, self.embed_dim))     
                    
        text_hidden_states = text_hidden_states.reshape(batch_size, self.num_queries*self.embed_dim)  
        text_risk_feat = self.text_feat_projection(text_hidden_states)
        imgtext_risk_feat = self.imgtext_fuse_projection(torch.concat([img_risk_feat, text_risk_feat], dim=1),)
        fused_risk_feat = self.clin_fuse_projection(torch.concat([imgtext_risk_feat, clin_risk_feat], dim=1),)
        # fused_risk_feat = clin_risk_feat
        risk_score = self.risk_predictor(fused_risk_feat.reshape(batch_size, -1))

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

    @torch.no_grad()
    def risk_predict(self, batch, device):
        risk_score = self.stage3_step(batch, device, return_score=True)
        return risk_score        
        
    def risk_predict_grad(self, batch, device='cuda'):
        batch_size = batch["clin_feat"].size(0)
        clin_feat = batch["clin_feat"].to(device, non_blocking=True)
        clin_risk_feat = self.clin_risk_feat_encoder(clin_feat)
        
        risk_feat, global_feat, features = self.img_feat_encode(batch, device, mean_pool=True)
    
        # img + MRE + SSE (text_align)
        image_sentence_embed = self.region_encode(batch, global_feat, risk_feat, features, device)
        img_risk_feat = risk_feat


        text_hidden_states = self.feature_space_transformation_nn(image_sentence_embed.view(batch_size * self.num_queries, self.embed_dim))     
                    
        text_hidden_states = text_hidden_states.reshape(batch_size, self.num_queries*self.embed_dim)  
        text_risk_feat = self.text_feat_projection(text_hidden_states)
        imgtext_risk_feat = self.imgtext_fuse_projection(torch.concat([img_risk_feat, text_risk_feat], dim=1),)
        fused_risk_feat = self.clin_fuse_projection(torch.concat([imgtext_risk_feat, clin_risk_feat], dim=1),)
        # fused_risk_feat = clin_risk_feat
        risk_score = self.risk_predictor(fused_risk_feat.reshape(batch_size, -1))
        return risk_score        
if __name__ == '__main__':
    model = CrossAlignModel()

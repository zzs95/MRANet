import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import spacy
import numpy as np
import pandas as pd
import torch
from dataset.constants import REPORT_KEYS, COVID_REGIONS
from dataset.create_image_report_dataloader import get_data_loaders
from models.risk_model.region_align_model import CrossAlignModel
from configs.region_align_model_config import *
from path_datasets_and_weights import path_runs
from utils.utils import write_config, seed_everything
from utils.file_and_folder_operations import *
from evaluate_utils.sents_reports_utils import get_generated_reports

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)
seed_everything(SEED)
# python -m spacy download en_core_web_trf
sentence_tokenizer = spacy.load("en_core_web_trf")


class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al. 
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''
    def __init__(self, model, target_layers, dim=1280, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers
        self.dim = dim
        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)
        
        self.activations = []
        self.grads = []
        
    def forward_hook(self, module, input, output):
        self.activations.append(input[0])
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_input[0].detach())
        
    def calculate_cam(self, model_input):
        self.model.eval()
        
        # forward
        y_c = self.model.risk_predict_grad(model_input)

        # backward
        self.model.zero_grad()
        y_c.backward()
        
        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()
        
        # calculate weights
        weights = np.mean(grads.reshape(-1, self.dim), axis=1)
        activations = activations.reshape(-1, self.dim)
        cam = (weights[:,None] * activations).sum(axis=1)
        # cam = np.maximum(cam, 0) # ReLU
        # cam = cam - cam.min()
        # cam = cam / cam.max()
        return cam

def gen_model(
    model,
    train_dl,
    gpt_tokenizer,
    generated_sentences_and_reports_folder_path,
    device,
    cfg,
    set_name
):
    model.eval()
    gen_text = {
        "idxs": [],
        "generated_reports": [],
        "reference_reports": [],
        "time_label": [],
        'event_label': [],
        'risk_score': []
    }
    key_list = list(COVID_REGIONS.keys()) + ['IMPRESSION']
    for k in key_list:
        gen_text['generated_sentence_'+k] = []
        gen_text['reference_sentence_'+k] = []
    for i, k in enumerate(COVID_REGIONS.keys()):
        gen_text['cams_'+k] = []
    for num_batch, batch in enumerate(train_dl):
 
        print(num_batch)
        torch.cuda.empty_cache()
        batch_size = batch['clin_feat'].shape[0]
        reference_sentences = batch["reference_sentences"]
        reference_reports = batch["reference_reports"]       
        with torch.no_grad(): 
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.generate_text(batch, device,
                            max_length=MAX_NUM_TOKENS_GENERATE,
                            num_beams=NUM_BEAMS,
                            early_stopping=True,)
                risk_score = model.risk_predict(batch, device,)
        generated_sents_for_selected_regions = gpt_tokenizer.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        reference_sents_for_selected_regions = [j for i in reference_sentences for j in i]
        selected_regions = np.ones([batch_size, len(REPORT_KEYS)])
        generated_reports, removed_similar_generated_sentences = get_generated_reports(
            generated_sents_for_selected_regions,
            selected_regions,
            sentence_tokenizer ,
            BERTSCORE_SIMILARITY_THRESHOLD
        )
        
        cams = []
        for i, k in enumerate(COVID_REGIONS.keys()):
            grad_cam = GradCAM(model, model.region_visual_embedder[i].fc5)
            cam = grad_cam.calculate_cam(batch)
            cams.append(cam)
        cams_max =  np.abs(np.concatenate(cams)).max()
        
        for i, k in enumerate(COVID_REGIONS.keys()):
            cams[i] = cams[i] / cams_max
            cams[i] = np.maximum(cams[i], 0) # ReLU
            print(cams[i])
       
        for j in range(batch_size):
            for i, k in enumerate(key_list):
                gen_text["generated_sentence_"+k].append(generated_sents_for_selected_regions[j*5+i])
                if set_name == 'brown':
                    gen_text["reference_sentence_"+k].append(reference_sents_for_selected_regions[j*5+i])
                elif set_name == 'penn':
                    gen_text["reference_sentence_"+k].append([])
        for i, k in enumerate(COVID_REGIONS.keys()):
            gen_text["cams_"+k].append(cams[i].tolist())
                
        gen_text["generated_reports"].extend(generated_reports)
        gen_text["reference_reports"].extend(reference_reports)
        gen_text["idxs"].extend(batch["idxs"].numpy())
        gen_text["time_label"].extend(batch["time_label"].cpu().numpy())
        gen_text["event_label"].extend(batch["event_label"].cpu().numpy())
        gen_text["risk_score"].extend(risk_score[:,0].cpu().numpy())

        log.info("Test loader: generating sentences/reports... DONE.")

    pd.DataFrame.from_dict(gen_text).to_excel(os.path.join(generated_sentences_and_reports_folder_path, set_name+'_text_cam.xlsx'))
    return None

def get_model(device):
    model = CrossAlignModel(stage='pred')
    CHECKPOINT = './checkpoints/fuse_risk_model.pt'
    sur_checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    log.info("Load checkpoint:"+CHECKPOINT)
    CHECKPOINT = './checkpoints/region_align_model.pt'
    gen_checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    log.info("Load checkpoint:"+CHECKPOINT)
    
    # model.load_state_dict(checkpoint["model"])
    
    # load survival params
    from collections import OrderedDict
    new_state_dict = OrderedDict()        
    for k, v in sur_checkpoint["model"].items():
        new_state_dict[k] = v
    # load generate text params
    for k, v in gen_checkpoint["model"].items():
        if 'language_model' in k:
            new_state_dict[k] = v    
        if 'gatortron_projection' in k:
            new_state_dict[k] = v  
    model.load_state_dict(new_state_dict)
    model.to(device, non_blocking=True)
    
    del sur_checkpoint
    del gen_checkpoint
    return model

def create_run_folder():
    run_folder_path = os.path.join(path_runs+'_vis_results', 'img_MRE_SSE_text' )
    generated_sentences_and_reports_folder_path = run_folder_path
    
    if os.path.exists(run_folder_path):
        log.info(f"Folder to save run {RUN} already exists at {run_folder_path}. ")
        if not RESUME_TRAINING:
            log.info(f"Delete the folder {run_folder_path}.")
    maybe_mkdir_p(run_folder_path)
    log.info(f"Run {RUN} folder created at {run_folder_path}.")
    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        'BATCH_SIZE': BATCH_SIZE,
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "BERTSCORE_SIMILARITY_THRESHOLD": BERTSCORE_SIMILARITY_THRESHOLD,
    }
    return config_file_path, config_parameters, generated_sentences_and_reports_folder_path
BATCH_SIZE = 1
def main():
    (config_file_path, config_parameters, generated_sentences_and_reports_folder_path) = create_run_folder()
    train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], image_input_size=IMAGE_INPUT_SIZE, return_all=False, 
                                                                                           random_state_i=config_parameters['SEED'])
    config_parameters["Brown TEST total NUM"] = len(test_loader.dataset)
    config_parameters["Brown TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # model, text_sur_predictor, text_encoder = get_model(device)
    model = get_model(device)
    log.info("Starting generating!")
    gen_model(
        model=model,
        train_dl=test_loader,
        gpt_tokenizer=gpt_tokenizer,
        generated_sentences_and_reports_folder_path=generated_sentences_and_reports_folder_path,
        device=device,
        cfg=config_parameters,
        set_name='brown'
    )

    test_loader, gpt_tokenizer = get_data_loaders(setname='penn', batch_size=config_parameters['BATCH_SIZE'], image_input_size=IMAGE_INPUT_SIZE, return_all=True, 
                                                                                           random_state_i=config_parameters['SEED'])
    config_parameters["Penn TEST total NUM"] = len(test_loader.dataset)
    config_parameters["Penn TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']
    write_config(config_file_path, config_parameters)
    gen_model(
        model=model,
        train_dl=test_loader,
        gpt_tokenizer=gpt_tokenizer,
        generated_sentences_and_reports_folder_path=generated_sentences_and_reports_folder_path,
        device=device,
        cfg=config_parameters,
        set_name='penn'
    )

if __name__ == "__main__":
    main()

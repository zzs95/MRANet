import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import spacy
import numpy as np
import pandas as pd
import torch
from dataset.constants import REPORT_KEYS
from dataset.create_image_report_dataloader import get_data_loaders
from models.risk_model.region_align_model import CrossAlignModel
from configs.region_model_align_risk_config import *
from path_datasets_and_weights import path_runs
from utils.utils import write_config, seed_everything
from utils.file_and_folder_operations import *
from evaluate_utils.sents_reports_utils import update_gen_and_ref_sentences_for_regions, update_gen_sentences_with_corresponding_regions, get_generated_reports, update_num_generated_sentences_per_image
from evaluate_utils.write_utils import write_sentences_and_reports_to_file_for_test_set

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)
seed_everything(SEED)
# python -m spacy download en_core_web_trf
sentence_tokenizer = spacy.load("en_core_web_trf")
def gen_model(
    model,
    train_dl,
    gpt_tokenizer,
    generated_sentences_and_reports_folder_path,
    device,
    cfg,
    set_name
):
    gen_and_ref_sentences = {
        "generated_sentences": [],
        "reference_sentences": [],
        "num_generated_sentences_per_image": []
    }
    gen_and_ref_reports = {
        "idxs": [],
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }
    for region_index in REPORT_KEYS:
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }
    gen_sentences_with_corresponding_regions = []
    model.eval()
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
        gen_and_ref_sentences["generated_sentences"].extend(generated_sents_for_selected_regions)
        gen_and_ref_sentences["reference_sentences"].extend(reference_sents_for_selected_regions)
        gen_and_ref_reports["idxs"].extend(batch["idxs"].numpy())
        gen_and_ref_reports["generated_reports"].extend(generated_reports)
        gen_and_ref_reports["reference_reports"].extend(reference_reports)
        gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

        update_gen_and_ref_sentences_for_regions(gen_and_ref_sentences, generated_sents_for_selected_regions, reference_sents_for_selected_regions, selected_regions)
        update_num_generated_sentences_per_image(gen_and_ref_sentences, selected_regions)

        # if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
        update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)
        # break
        
        del batch
        del reference_sentences
        del reference_reports
        del generated_reports
        del output
        del selected_regions
        del generated_sents_for_selected_regions
        del reference_sents_for_selected_regions
        del removed_similar_generated_sentences
    log.info("Test loader: generating sentences/reports... DONE.")

    for i in range(len(gen_and_ref_reports["generated_reports"])):
        gen_and_ref_reports["generated_reports"][i] = 'Findings: '+ gen_and_ref_reports["generated_reports"][i] 
    
    write_sentences_and_reports_to_file_for_test_set(
        gen_and_ref_sentences,
        gen_and_ref_reports,
        gen_sentences_with_corresponding_regions,
        generated_sentences_and_reports_folder_path,
        set_name,
    )
    
    out_dict = {'indx': gen_and_ref_reports['idxs'], 'report':gen_and_ref_reports['generated_reports']}
    pd.DataFrame.from_dict(out_dict).to_excel(os.path.join(generated_sentences_and_reports_folder_path, set_name+'_text.xlsx'))
    return None

def get_model(device):
    model = CrossAlignModel()
    CHECKPOINT = path_runs + '/align_model/run_2/checkpoints/epoch_14999.pt'
                
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    log.info("Load checkpoint:"+CHECKPOINT)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    del checkpoint
    return model

def create_run_folder():
    run_folder_path = os.path.join(path_runs+'_results', 'gen_text' )
    generated_sentences_and_reports_folder_path = run_folder_path
    
    if os.path.exists(run_folder_path):
        log.info(f"Folder to save run {RUN} already exists at {run_folder_path}. ")
        if not RESUME_TRAINING:
            log.info(f"Delete the folder {run_folder_path}.")
            # if local_rank == 0:           
                # shutil.rmtree(run_folder_path)
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
BATCH_SIZE = 6
def main():
    (config_file_path, config_parameters, generated_sentences_and_reports_folder_path) = create_run_folder()
    train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], image_input_size=IMAGE_INPUT_SIZE, return_all=False, 
                                                                                           random_state_i=config_parameters['SEED'])
    config_parameters["Brown TEST total NUM"] = len(test_loader.dataset)
    config_parameters["Brown TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
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

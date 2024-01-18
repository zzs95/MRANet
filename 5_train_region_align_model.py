import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm
from evaluate_utils.write_utils import write_all_losses_and_scores_to_tensorboard_feat
from dataset.create_image_report_dataloader import get_data_loaders
# from models.feat_text_generator.evaluate import evaluate_model
from models.risk_model.region_align_model import CrossAlignModel
from configs.region_model_align_risk_config import *
from path_datasets_and_weights import path_runs
from utils.utils import write_config, seed_everything
from utils.file_and_folder_operations import *
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 
parser = argparse.ArgumentParser(description='Network Parser')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int) 
args = parser.parse_args()
local_rank = args.local_rank

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device=torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=timedelta(seconds=18000))
if not DDP:
    local_rank = 0 # DP
# seed_everything(SEED)

def call_optimization(model, warmup_epochs=None, learning_rate=None, learning_rate_start=None, learning_rate_end=None, weight_decay=None):
    # params = model.parameters()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    warmup_steps =  warmup_epochs 
    scheduler = MultiStepLR(optimizer, milestones=[warmup_steps, warmup_steps * 2], gamma=0.5, verbose=True)
    return optimizer, scheduler
    

def train_model(
    model,
    train_dl,
    val_dl,
    scaler,
    train_sampler,
    current_epoch,
    overall_steps_taken,
    lowest_val_loss,
    checkpoints_folder_path,
    gpt_tokenizer,
    generated_sentences_and_reports_folder_path,
    writer,
    log_file,
    device,
    cfg
):
    run_params = cfg
    run_params["epochs"] = cfg['EPOCHS']
    run_params["checkpoints_folder_path"] = checkpoints_folder_path
    run_params["lowest_val_loss"] = lowest_val_loss
    run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
    run_params["overall_steps_taken"] = overall_steps_taken  # for logging to tensorboard
    run_params["log_file"] = log_file  # for logging error messages (e.g. OOM)
    train_iters_per_epoch = len(train_dl)  
    run_params["train_iters_per_epoch"] = train_iters_per_epoch
    model.train()
    # for gradient accumulation
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // cfg['BATCH_SIZE']
    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False
    optimizer, lr_scheduler = call_optimization(model, warmup_epochs=stage2_warmup_epochs, weight_decay=stage2_weight_decay, learning_rate=stage2_learning_rate, learning_rate_start=stage2_learning_rate_start, learning_rate_end=stage2_learning_rate_end)
    if RESUME_TRAINING:
        optimizer.load_state_dict(run_params['checkpoint']['optimizer'])
        lr_scheduler.load_state_dict(run_params['checkpoint']['lr_scheduler'])
        del run_params['checkpoint']
    for epoch in range(current_epoch, cfg['EPOCHS']):
        if run_params['DDP']:
            train_sampler.set_epoch(epoch) # DDP

        run_params["lowest_val_loss"] = np.inf
        run_params["epoch"] = epoch
        train_losses_dict = {
            'total_loss':0.0,
            'loss': 0.0, 'lm_img_loss':0.0, 'lm_text_loss':0.0,  'surv_loss':0.0
        }

        run_params["steps_taken"] = 0  # to know when to evaluate model during epoch and to normalize losses
        for num_batch, batch in enumerate(train_dl):
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    losses_dict = model(batch, device)
                    total_loss = losses_dict['loss']
                    scaler.scale(total_loss).backward()

            except RuntimeError as e:  # out of memory error
                log.info(f"Error: {e}")
                if "out of memory" in str(e):
                    with open(run_params["log_file"], "a") as f:
                        f.write("Training:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e
            # torch.cuda.empty_cache()
            if oom:
                # free up memory
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad56
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                oom = False
                continue

            if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # lr_scheduler.step()    
                
                # torch.cuda.empty_cache()
            train_losses_dict['total_loss'] += total_loss.item() 

            # dicts are insertion ordered since Python 3.7
            for loss_type in losses_dict.keys():
                loss = losses_dict[loss_type].mean()
                train_losses_dict[loss_type] += loss.item() 

            run_params["steps_taken"] += 1
            run_params["overall_steps_taken"] += 1
        lr_scheduler.step() 
        if local_rank == 0:
            for loss_type in train_losses_dict:
                train_losses_dict[loss_type] /= run_params["steps_taken"]
            log.info(f"Training epoch {epoch}! Loss {train_losses_dict['total_loss']}!\n")
            write_all_losses_and_scores_to_tensorboard_feat(
                writer,
                run_params["overall_steps_taken"],
                train_losses_dict,
                None,
                float(optimizer.param_groups[0]["lr"])
            )
            # reset values for the next evaluation
            for loss_type in train_losses_dict:
                train_losses_dict[loss_type] = 0.0
            
            # if (run_params["epoch"]+1) % run_params['EVALUATE_EVERY_K_EPOCHS'] == 0:  
            if run_params["epoch"] > 0 and (run_params["epoch"] + 1) % run_params['EVALUATE_EVERY_K_EPOCHS'] == 0:  
                log.info(f"Evaluating at epoch {run_params['epoch']}!")
                evaluate_model(
                    model,
                    train_dl,
                    lr_scheduler,
                    optimizer,
                    scaler,
                    writer,
                    gpt_tokenizer,
                    run_params,
                    generated_sentences_and_reports_folder_path,
                    device
                )
                # set the model back to training
                model.train()
                optimizer.zero_grad()
            run_params["steps_taken"] = 0
            
    dist.destroy_process_group()
    log.info("Finished training!")
    log.info(f"Lowest overall val loss: {run_params['lowest_val_loss']:.3f} at epoch {run_params['best_epoch']}")
    return None

def evaluate_model(model, val_dl, lr_scheduler, optimizer, scaler, writer, tokenizer, run_params, generated_sentences_and_reports_folder_path, device):
    model.eval()
    epoch = run_params["epoch"]
    # stage = run_params["stage"]
    # local_stage = run_params["local_stage"]
    steps_taken = run_params["steps_taken"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    total_val_loss = 0
    save_path = os.path.join(run_params["checkpoints_folder_path"], f"epoch_{epoch}.pt")
    if run_params['MULTI_GPU']:
        checkpoint = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "current_epoch": epoch,
            "overall_steps_taken": overall_steps_taken,
            "lowest_val_loss": total_val_loss,
        }
    else:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "current_epoch": epoch,
            "overall_steps_taken": overall_steps_taken,
            "lowest_val_loss": total_val_loss,
        }

    torch.save(checkpoint, save_path)

def get_model(device):
    model = CrossAlignModel(stage='stage_2')
    if not RESUME_TRAINING:
        checkpoint = torch.load('./checkpoints/image_risk_model.pt', map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    model_init_state_dict = model.state_dict()
    modules = np.unique([k for k in model_init_state_dict.keys()])
    stage1_modules = np.unique([k for k in checkpoint["model"].keys()])
    load_modules = np.intersect1d(modules, stage1_modules).tolist()
    load_module_names = np.unique([k.split('.')[0] for k in load_modules]).tolist()
    log.info("Load "+ str(load_module_names))
    from collections import OrderedDict
    new_state_dict = OrderedDict()        
    for k, v in model_init_state_dict.items():
        if k in load_modules:
            new_state_dict[k] = checkpoint["model"][k]
        else:
            new_state_dict[k] = model_init_state_dict[k]
    model.load_state_dict(new_state_dict)
    # checkpoint = None
    model.to(device, non_blocking=True)
    return model, checkpoint

def create_run_folder():
    run_folder_path = os.path.join(path_runs, 'align_model', f"run_{RUN}")
    checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    generated_sentences_and_reports_folder_path = os.path.join(run_folder_path, "generated_sentences_and_reports")
    generated_sentences_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences")
    generated_reports_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_reports")
    log_file = os.path.join(run_folder_path, "log_file")
    
    if os.path.exists(run_folder_path):
        log.info(f"Folder to save run {RUN} already exists at {run_folder_path}. ")
        if not RESUME_TRAINING:
            log.info(f"Delete the folder {run_folder_path}.")
            if local_rank == 0:           
                shutil.rmtree(run_folder_path)
    maybe_mkdir_p(run_folder_path)
    maybe_mkdir_p(checkpoints_folder_path)
    maybe_mkdir_p(tensorboard_folder_path)
    maybe_mkdir_p(generated_sentences_and_reports_folder_path)
    maybe_mkdir_p(generated_sentences_folder_path)
    maybe_mkdir_p(generated_reports_folder_path)
    log.info(f"Run {RUN} folder created at {run_folder_path}.")
        
    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        'BATCH_SIZE': BATCH_SIZE,
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "EVALUATE_EVERY_K_EPOCHS": EVALUATE_EVERY_K_EPOCHS,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "BERTSCORE_SIMILARITY_THRESHOLD": BERTSCORE_SIMILARITY_THRESHOLD,
        "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
        "NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
        "NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION": NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
        "MULTI_GPU": MULTI_GPU,
        "DDP": DDP
    }

    return checkpoints_folder_path, tensorboard_folder_path, config_file_path, config_parameters, generated_sentences_and_reports_folder_path, log_file
def main():
    torch.cuda.set_device(local_rank)
    (checkpoints_folder_path, tensorboard_folder_path, config_file_path, config_parameters, generated_sentences_and_reports_folder_path, log_file) = create_run_folder()
    seed_everything(config_parameters['SEED'])
    train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], DDP=DDP, image_input_size=IMAGE_INPUT_SIZE, is_token=True, random_state_i=config_parameters['SEED'])
    config_parameters["TRAIN total NUM"] = len(train_loader.dataset)
    config_parameters["TRAIN index NUMs"] = train_loader.dataset.tokenized_dataset['index']
    config_parameters["VAL total NUM"] = len(val_loader.dataset)
    config_parameters["VAL index NUMs"] = val_loader.dataset.tokenized_dataset['index']
    config_parameters["TEST total NUM"] = len(test_loader.dataset)
    config_parameters["TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']
    write_config(config_file_path, config_parameters)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model, checkpoint = get_model(device)
    scaler = torch.cuda.amp.GradScaler()

    current_epoch = 0
    overall_steps_taken = 0
    lowest_val_loss = np.inf
    
    if RESUME_TRAINING:
        log.info("Resume training from RUN_"+str(RUN))
        # model.load_state_dict(checkpoint["model"])
        scaler.load_state_dict(checkpoint["scaler"])
        current_epoch = checkpoint["current_epoch"]
        overall_steps_taken = checkpoint["overall_steps_taken"]
        lowest_val_loss = checkpoint["lowest_val_loss"]
        config_parameters['checkpoint'] = checkpoint
    # del checkpoint
    torch.cuda.empty_cache()

    if MULTI_GPU:
        if config_parameters['DDP']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda() # DDP
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # DDP
        else:
            model = nn.DataParallel(model) # DP
            
    writer = SummaryWriter(log_dir=tensorboard_folder_path)

    log.info("Starting training!")
    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        scaler=scaler,
        train_sampler=train_sampler,
        current_epoch=current_epoch,
        overall_steps_taken=overall_steps_taken,
        lowest_val_loss=lowest_val_loss,
        checkpoints_folder_path=checkpoints_folder_path,
        gpt_tokenizer=gpt_tokenizer,
        generated_sentences_and_reports_folder_path=generated_sentences_and_reports_folder_path,
        writer=writer,
        log_file=log_file,
        device=device,
        cfg=config_parameters
    )


if __name__ == "__main__":
    main()

    # python -m torch.distributed.launch --nproc_per_node=2 5_train_region_align_modal.py 
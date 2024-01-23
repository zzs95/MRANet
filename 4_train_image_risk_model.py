import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
# import torch.nn as nn
# from tqdm import tqdm
from evaluate_utils.write_utils import write_all_losses_and_scores_to_tensorboard_feat
from dataset.create_image_report_dataloader import get_data_loaders
from models.risk_model.single_modal import SingleModalModel
from configs.risk_prediction_config import *
from path_datasets_and_weights import path_runs
from utils.utils import write_config, seed_everything
from utils.file_and_folder_operations import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

def call_optimization(model, max_epochs=None, warmup_epochs=None, learning_rate=None, learning_rate_start=None, learning_rate_end=None, weight_decay=None):
    # params = model.parameters()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    warmup_steps =  warmup_epochs 
    total_steps = max_epochs
    PATIENCE_LR_SCHEDULER = 2  # number of evaluations (PATIENCE_epoch = PATIENCE_LR_SCHEDULER*EVALUATE_EVERY_K_EPOCHS) to wait for val loss to reduce before lr is reduced
    THRESHOLD_LR_SCHEDULER = 1e-4
    FACTOR_LR_SCHEDULER = 0.5
    COOLDOWN_LR_SCHEDULER = 5
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    
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
    gpt_tokenizer,
    writer,
    device,
    cfg
):
    run_params = cfg
    run_params["epochs"] = cfg['EPOCHS']
    run_params["lowest_val_loss"] = lowest_val_loss
    run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
    run_params["overall_steps_taken"] = overall_steps_taken  # for logging to tensorboard
    model.train()
    optimizer, lr_scheduler = call_optimization(model, max_epochs=epochs, warmup_epochs=warmup_epochs, weight_decay=weight_decay, learning_rate=learning_rate, learning_rate_start=learning_rate_start, learning_rate_end=learning_rate_end)
    run_params["lowest_val_loss"] = np.inf
    for epoch in range(current_epoch, cfg['EPOCHS']):
        run_params["epoch"] = epoch
        train_losses_dict = {
            "total_loss": 0.0, 'surv_loss': 0.0, 'loss': 0.0
        }

        run_params["steps_taken"] = 0  # to know when to evaluate model during epoch and to normalize losses
        for num_batch, batch in enumerate(train_dl):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                total_loss, losses_dict = model(run_params, batch, device)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            train_losses_dict['total_loss'] += total_loss.item() 

            for loss_type in losses_dict.keys():
                loss = losses_dict[loss_type].mean()
                train_losses_dict[loss_type] += loss.item() 

            run_params["steps_taken"] += 1
            run_params["overall_steps_taken"] += 1

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
        
        if run_params["epoch"] > 0 and (run_params["epoch"] + 1) % run_params['EVALUATE_EVERY_K_EPOCHS'] == 0:  
            log.info(f"Evaluating at epoch {run_params['epoch']}!")
            evaluate_model(
                model,
                val_dl,
                lr_scheduler,
                optimizer,
                scaler,
                writer,
                run_params,
                device
            )
            # set the model back to training
            model.train()
        run_params["steps_taken"] = 0
           
    log.info("Finished training!")
    log.info(f"Lowest overall val loss: {run_params['lowest_val_loss']:.3f} at epoch {run_params['best_epoch']}")
    return None

def evaluate_model(model, val_dl, lr_scheduler, optimizer, scaler, writer, run_params, device):
    model.eval()
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    val_losses_dict = {
        "total_loss": 0.0, 
        'loss': 0.0, 'surv_loss': 0.0, 
    }
    # evaluate on loss
    with torch.no_grad():
        steps_taken = 0
        for num_batch, batch in enumerate(val_dl):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                total_loss, losses_dict = model(run_params, batch, device)
            val_losses_dict['total_loss'] += total_loss.item() 

            for loss_type in losses_dict.keys():
                loss = losses_dict[loss_type].mean()
                val_losses_dict[loss_type] += loss.item() 
            steps_taken += 1
    for loss_type in val_losses_dict.keys():
        val_losses_dict[loss_type] /= steps_taken

    write_all_losses_and_scores_to_tensorboard_feat(
        writer,
        overall_steps_taken,
        val_losses_dict=val_losses_dict,)
    total_val_loss = val_losses_dict["total_loss"]
    lr_scheduler.step(total_val_loss)
    
    if total_val_loss < run_params["lowest_val_loss"]:
        run_params["lowest_val_loss"] = total_val_loss
        run_params["best_epoch"] = epoch
        save_path = os.path.join(run_params["checkpoints_folder_path"], f"epoch_{epoch}.pt")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "current_epoch": epoch,
            "overall_steps_taken": overall_steps_taken,
        }
        torch.save(checkpoint, save_path)

def get_model(device):
    model = SingleModalModel()
    if CHECKPOINT == None:
        checkpoint = CHECKPOINT
    else:
        checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))

    if checkpoint:
        log.info("Load checkpoint:"+CHECKPOINT)
        model.load_state_dict(checkpoint["model"])
        
    model.to(device, non_blocking=True)
    return model

def create_run_folder():
    run_folder_path = os.path.join(path_runs, exp_name, f"run_{RUN}")
    checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    
    if os.path.exists(run_folder_path):
        log.info(f"Folder to save run {RUN} already exists at {run_folder_path}. ")
        # if not RESUME_TRAINING:
        #     log.info(f"Delete the folder {run_folder_path}.")
        #     if local_rank == 0:           
        #         shutil.rmtree(run_folder_path)
    maybe_mkdir_p(run_folder_path)
    maybe_mkdir_p(checkpoints_folder_path)
    maybe_mkdir_p(tensorboard_folder_path)
    log.info(f"Run {RUN} folder created at {run_folder_path}.")
        
    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "EVALUATE_EVERY_K_EPOCHS": EVALUATE_EVERY_K_EPOCHS,
        "EPOCHS": epochs,
        "MULTI_GPU": MULTI_GPU,
        "DDP": DDP,
        "BATCH_SIZE": BATCH_SIZE,
        "checkpoints_folder_path": checkpoints_folder_path
    }
    return run_folder_path, tensorboard_folder_path, config_file_path, config_parameters

def main():
    (run_folder_path, tensorboard_folder_path, config_file_path, config_parameters) = create_run_folder(SEED)
    seed_everything(config_parameters['SEED'])
    data_seed = SEED
    train_loader, val_loader, test_loader, gpt_tokenizer, train_sampler = get_data_loaders(setname='brown', batch_size=config_parameters['BATCH_SIZE'], 
                                                                                           image_input_size=IMAGE_INPUT_SIZE, is_token=False, random_state_i=data_seed,
                                                                                           worker_seed=config_parameters['SEED'], )
    config_parameters["TRAIN total NUM"] = len(train_loader.dataset)
    config_parameters["TRAIN index NUMs"] = train_loader.dataset.tokenized_dataset['index']
    config_parameters["VAL total NUM"] = len(val_loader.dataset)
    config_parameters["VAL index NUMs"] = val_loader.dataset.tokenized_dataset['index']
    config_parameters["TEST total NUM"] = len(test_loader.dataset)
    config_parameters["TEST index NUMs"] = test_loader.dataset.tokenized_dataset['index']
    write_config(config_file_path, config_parameters)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = get_model(device)
    scaler = torch.cuda.amp.GradScaler()

    current_epoch = 0
    overall_steps_taken = 0
    lowest_val_loss = np.inf

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
        gpt_tokenizer=gpt_tokenizer,
        writer=writer,
        device=device,
        cfg=config_parameters
    )
    best_epoch = config_parameters["best_epoch"]
    
    best_val_checkpoint = os.path.join(config_parameters["checkpoints_folder_path"], f"epoch_{best_epoch}.pt")
    checkpoint = torch.load(best_val_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    event_labels = []
    time_labels = []
    risk_scores = []
    for num_batch, batch in enumerate(test_loader):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                risks = model.risk_predict(config_parameters, batch, device)
                time_labels.append(batch['time_label'].numpy())
                event_labels.append(batch['event_label'].numpy())
                risk_scores.append(risks.cpu().detach().numpy()[:,0])
            
    event_labels = np.concatenate(event_labels).astype(bool)
    time_labels = np.concatenate(time_labels)
    risk_scores = np.concatenate(risk_scores)
    test_df = test_loader.dataset.tokenized_dataset.to_pandas()
    test_df['risk_score'] = risk_scores
    test_df['time_label'] = time_labels
    test_df['event_label'] = event_labels
    test_df.to_excel(join(run_folder_path, 'brown_test.xlsx'))
        
    test_loader, gpt_tokenizer = get_data_loaders(setname='penn', return_all=True, batch_size=config_parameters['BATCH_SIZE'], 
                                                       image_input_size=IMAGE_INPUT_SIZE, is_token=False, random_state_i=config_parameters['SEED'])
    config_parameters["Penn total NUM"] = len(test_loader.dataset)
    write_config(config_file_path, config_parameters)
    event_labels = []
    time_labels = []
    risk_scores = []
    for num_batch, batch in enumerate(test_loader):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                risks = model.risk_predict(config_parameters, batch, device)
                time_labels.append(batch['time_label'].numpy())
                event_labels.append(batch['event_label'].numpy())
                risk_scores.append(risks.cpu().detach().numpy()[:,0])
            
    event_labels = np.concatenate(event_labels).astype(bool)
    time_labels = np.concatenate(time_labels)
    risk_scores = np.concatenate(risk_scores)
    risk_scores = np.nan_to_num(risk_scores)
    test_df = test_loader.dataset.tokenized_dataset.to_pandas()
    test_df['risk_score'] = risk_scores
    test_df['time_label'] = time_labels
    test_df['event_label'] = event_labels
    test_df.to_excel(join(run_folder_path, 'penn_test.xlsx'))
    
if __name__ == "__main__":
    exp_name = 'image_risk_model'
    out_folder = os.path.join(path_runs, exp_name, f"run_{RUN}")
    
    main()

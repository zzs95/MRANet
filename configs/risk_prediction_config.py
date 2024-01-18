"""
Define configurations for training run of full model.

If PRETRAIN_WITHOUT_LM_MODEL = True, then only the object detector and the 2 binary classifiers are trained in the full model,
with the language model (as the last component) being fully excluded from the model architecture.
This setting is for pre-training the 2 binary classifiers (together with the object detector),
since it's assumed that the object detector was already trained separately in object_detector/training_script_object_detector.py

If PRETRAIN_WITHOUT_LM_MODEL = False, then the full model is trained end-to-end.

Ideally, the training should go like this:

(1) Object detector training:
    - see src/object_detector/training_script_object_detector.py

(2) Object detector + binary classifiers training:
    - load best object detector weights from step (1) into the object detector in the __init__ method in src/full_model/report_generation_model.py
    - set PRETRAIN_WITHOUT_LM_MODEL = True in this file
    - make sure that in the main function of src/full_model/train_full_model.py,
    no other weights are loaded into the instantiated ReportGenerationModel (i.e. make sure that line 567 is commented out)
    - pre-train full model without language model with src/full_model/train_full_model.py

(3) Full model training:
    - uncomment lines that load object detector weights in the __init__ method (since those weights will be overwritten anyway)
    - set PRETRAIN_WITHOUT_LM_MODEL = False in this file
    - load best pre-trained full model weights from step (2) in src/full_model/train_full_model.py
    by specifying them in checkpoint = torch.load(...) in line 567
    - train full model with src/full_model/train_full_model.py
"""
RUN = 1
RUN_COMMENT = """Enter a comment here."""
IMAGE_INPUT_SIZE = 224

EVALUATE_EVERY_K_EPOCHS = 10
SEED = 0
CHECKPOINT = None
# MULTI_GPU = True
MULTI_GPU = False
DDP = True
# DDP = False
RESUME_TRAINING = False
# RESUME_TRAINING = True

epochs = 100
learning_rate, learning_rate_start, learning_rate_end, warmup_epochs, weight_decay  = 1e-4, 1e-5, 0, 150, 1e-6
# stage2_pretrain = '/media/brownradai/my4TB/covid_cxr_region_surv_runs/full_model/run_2/stage1/checkpoints/epoch_299.pt'
BATCH_SIZE = 1000
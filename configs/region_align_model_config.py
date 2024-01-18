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
RUN = 2
RUN_COMMENT = """Enter a comment here."""
SEED = 0
IMAGE_INPUT_SIZE = 224
BATCH_SIZE = 12 # 12
# EFFECTIVE_BATCH_SIZE = 64  # batch size after gradient accumulation
EFFECTIVE_BATCH_SIZE = 36  # batch size after gradient accumulation

EPOCHS = 15000 
# how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
# EVALUATE_EVERY_K_BATCHES should be divisible by ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
EVALUATE_EVERY_K_EPOCHS = 500
# EVALUATE_EVERY_K_BATCHES = 50
PATIENCE_LR_SCHEDULER = 10  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
NUM_BEAMS = 4
# MAX_NUM_TOKENS_GENERATE is set arbitrarily to 300. Most generated sentences have at most 60 tokens,
# so this is just an arbitrary threshold that will never be reached if the language model is not completely untrained (i.e. produces gibberish)
MAX_NUM_TOKENS_GENERATE = 128
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated reports with their gt reference reports to a txt file
NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION = 100 # 100  # for evaluation of bleu, rouge-l and meteor
BERTSCORE_SIMILARITY_THRESHOLD = 0.9  # threshold for discarding generated sentences that are too similar
WEIGHT_OBJECT_DETECTOR_LOSS = 1
WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS = 5
WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS = 5
MULTI_GPU = True
# MULTI_GPU = False
DDP = True
# DDP = False
RESUME_TRAINING = False
# RESUME_TRAINING = True
CHECKPOINT = None
# CHECKPOINT = "/media/brownradai/my4TB/covid_cxr_region_surv_runs/cross_align_model_img_SSE/run_2/checkpoints/epoch_7599.pt"

stage2_learning_rate, stage2_learning_rate_start, stage2_learning_rate_end, stage2_warmup_epochs, stage2_weight_decay  = 1e-5, 1e-7, 0, 4000, 1e-6
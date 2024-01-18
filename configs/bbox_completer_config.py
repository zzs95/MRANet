# define configurations for training run
RUN = 'bbox_completer'
# comment can be useful to add additional information to run_config.txt file
RUN_COMMENT = """Enter comment here."""
SEED = 41
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.2
BATCH_SIZE = 2000
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 16
EPOCHS = 100
LR = 1e-3
EVALUATE_EVERY_K_STEPS = 4000  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced
THRESHOLD_LR_SCHEDULER = 1e-3
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
# CHECKPOINT = None
MULTI_GPU = False
path_dataset ='/media/brownradai/ssd_2t/covid_cxr/mimic-cxr-jpg-reports_full'
# path_dataset ='/media/brownradai/ssd_2t/covid_cxr/mimic-cxr-jpg-reports_200'

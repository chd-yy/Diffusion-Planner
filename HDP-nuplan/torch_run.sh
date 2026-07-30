export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH= # python path (e.g., "/home/xxx/anaconda3/envs/hdp_nuplan/bin/python")

# Set training data path
TRAIN_SET_PATH= # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH= # json list of processed data
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--diffusion_model_type "x_start" \
--diffusion_supervision_type "x_start" \
--batch_size 2048
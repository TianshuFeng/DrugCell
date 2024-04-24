


echo DrugCell SETTINGS
echo SETTINGS
# General Settings
export PROCS=10
export PPN=5
export WALLTIME=12:00:00
export NUM_ITERATIONS=5
export POPULATION_SIZE=98
# GA Settings
export GA_STRATEGY='mu_plus_lambda'
export OFFSPRING_PROPORTION=0.5
export MUT_PROB=0.8
export CX_PROB=0.2
export MUT_INDPB=0.5
export CX_INDPB=0.5
export TOURNSIZE=4
# Lambda Settings
# export CANDLE_CUDA_OFFSET=1
#export CANDLE_DATA_DIR=/tmp/rgnanaolivu
# Polaris Settings
# export QUEUE="debug"
# Polaris Settings
export QUEUE="prod"
export CANDLE_DATA_DIR=/home/rgnanaolivu/improve/DrugCell/data/

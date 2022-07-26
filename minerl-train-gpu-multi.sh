#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=12:00:00
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-mcrowley 

module load cuda cudnn 
source tensorflow/bin/activate
python ./minerl_pretrain_actor_critic_32_actions.py

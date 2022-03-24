#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="hp2"
#SBATCH --out=hp_tuning2.out
#SBATCH --error=hp_tuning2.err
#SBATCH --mail-user=<arner97@gmail.com>
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "GPUs allocated for the job: $SLURM_JOB_GPUS"

module purge
module load TensorFlow/2.4.1-fosscuda-2020b
module load tensorflow-probability/0.12.1-fosscuda-2020b

python run_hp_idun2.py

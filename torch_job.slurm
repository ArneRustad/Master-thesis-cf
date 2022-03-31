#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-imf #share-ie-imf
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="torch"
#SBATCH --out=torch_job.out
#SBATCH --error=torch_job.err
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
module load PyTorch/1.7.1-fosscuda-2020b 

python gen_data_other_synthesizers.py

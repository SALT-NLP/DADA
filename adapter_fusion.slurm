#!/bin/bash
#SBATCH -c 4 # Number of cores requested
#SBATCH -t 7-0:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH --gres=gpu:4 # 需要使用多少GPU，n是需要的数量
#SBATCH --mem=102400 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p seas_gpu # Partition to submit to
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion/log/%j.out # Standard out goes to this file
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion/log/%j.err # Standard err goes to this filehostname
#SBATCH --chdir=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion

#SBATCH --mail-user=yanchenliu@fas.harvard.edu # 把通知发送到哪一个邮箱

module load Anaconda
source activate adapter_tuning
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29510 adapter_fusion.py --job_id $SLURM_JOB_ID

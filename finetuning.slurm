#!/bin/bash
#SBATCH -c 1 # Number of cores requested
#SBATCH -t 7-00:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --mem=102400 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p gpu # Partition to submit to
#SBATCH --open-mode=append # Append when writing files
#SBATCH -o /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion/log/%j.out # Standard out goes to this file
#SBATCH -e /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion/log/%j.err # Standard err goes to this filehostname
#SBATCH --chdir=/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yanchenliu/AdapterFusion

#SBATCH --mail-user=yanchenliu@fas.harvard.edu # 把通知发送到哪一个邮箱

module load Anaconda
source activate adapter_tuning
python finetuning.py --task_name $1 --train_data $2
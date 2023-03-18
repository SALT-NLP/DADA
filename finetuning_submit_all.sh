export task_name=MNLI
declare -a train_datas=("MULTI" "AppE" "ChcE" "CollSgE" "IndE" "VALUE" "GLUE")

for train_data in "${train_datas[@]}"
do
    sbatch finetuning.slurm $task_name $train_data
done
export task_name=MNLI
declare -a rules=("bare_past_tense" "nomo_existential" "that_resultative_past_participle")

for rule in "${rules[@]}"
do
    sbatch adapter_tuning.slurm $task_name $rule
done
#!/usr/bin/env bash

MEM=25GB
num_gpus=48
gpu_partition=titanx-long
#gpu_partition=m40-long

wordkeep='0.7 0.8 0.9 0.95'
#wordkeep='1.0'
randtype='Bernoulli Bernoulli-word Bernoulli-semantic Bernoulli-adversarial'
#randtype='None'
runtimes='1 2 3'
CMD="python train.py"
for _wordk in ${wordkeep[@]}; do
    for _randt in ${randtype[@]}; do
        for _time in ${runtimes[@]}; do
            tune_args="--word_keep=$_wordk --random_type=$_randt --time=$_time "
            commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.${_randt}.${_wordk}.${_time}")
        done
    done
done


num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
        echo "Starting job $jobid on gpu $gpuid"
        eval ${comm}
    done &
    j=$((j + 1))
done


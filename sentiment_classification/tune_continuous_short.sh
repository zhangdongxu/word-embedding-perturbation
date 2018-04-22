#!/usr/bin/env bash

MEM=25GB
num_gpus=18
gpu_partition=titanx-long
#gpu_partition=m40-long

fix_embedding="True"
noiseweight='0.001 0.01 0.1'
#randtype='Gaussian'
#randtype='Gaussian Gaussian-adversarial'
#randtype='Gaussian-adversarial'
randtype='Adversarial Gaussian-adversarial'
runtimes='1 2 3'
#CMD="python train.py --train_file=data/stsa.binary.phrases.train --dev_file=data/stsa.binary.dev --test_file=data/stsa.binary.test"
#CMD="python train.py --train_file=data/rt-polarity"
CMD="python train.py --train_file=data/custrev.all"
#CMD="python train.py --train_file=data/TREC.train.all --test_file=data/TREC.test.all"
for _fix in ${fix_embedding[@]}; do
    for _noisew in ${noiseweight[@]}; do
        for _randt in ${randtype[@]}; do
            for _time in ${runtimes[@]}; do
                tune_args="--train_emb=$_fix --noise_weight=$_noisew --random_type=$_randt "
                #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.sst2.${_randt}.${_fix}.${_noisew}.${_time}")
                #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.mr.${_randt}.${_fix}.${_noisew}.${_time}")
                commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.cr.${_randt}.${_fix}.${_noisew}.${_time}")
                #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.trec.${_randt}.${_fix}.${_noisew}.${_time}")
            done
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


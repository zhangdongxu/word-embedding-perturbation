#!/usr/bin/env bash

MEM=25GB
num_gpus=24
gpu_partition=titanx-short
#gpu_partition=titanx-short
#gpu_partition=m40-long

fix_embedding="True"
#wordkeep='0.9 0.8 0.7'
wordkeep='0.7 0.8 0.9 0.95'
#randtype='None Bernoulli Bernoulli-word Bernoulli-semantic Bernoulli-adversarial'
randtype='Replace'
topks='1 5'
runtimes='1 2 3'
#CMD="python train.py --train_file=data/stsa.binary.phrases.train --dev_file=data/stsa.binary.dev --test_file=data/stsa.binary.test"
#CMD="python train.py --train_file=data/rt-polarity"
#CMD="python train.py --train_file=data/TREC.train.all --test_file=data/TREC.test.all"
CMD="python train.py --train_file=data/custrev.all"
for _fix in ${fix_embedding[@]}; do
    for _wordk in ${wordkeep[@]}; do
        for _randt in ${randtype[@]}; do
            for _topk in ${topks[@]}; do
                for _time in ${runtimes[@]}; do
                    tune_args="--train_emb=$_fix --word_keep=$_wordk --topk=$_topk --random_type=$_randt "
                    #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.sst2.${_randt}.${_fix}.${_wordk}.${_topk}.${_time}")
                    #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.mr.${_randt}.${_fix}.${_wordk}.${_topk}.${_time}")
                    #commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.trec.${_randt}.${_fix}.${_wordk}.${_topk}.${_time}")
                    commands+=("srun --gres=gpu:1 --mem=$MEM --partition=${gpu_partition} ${CMD} $tune_args > log.cr.${_randt}.${_fix}.${_wordk}.${_topk}.${_time}")
                done
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


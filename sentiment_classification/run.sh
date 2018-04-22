nohup srun --gres=gpu:1 --mem=25GB --partition=titanx-long python train.py --train_file=data/stsa.binary.phrases.train --dev_file=data/stsa.binary.dev --test_file=data/stsa.binary.test --train_emb=True --word_keep=0.9 --noise_weight=0.01 --random_type=Adversarial > log.sst2.Adversarial.noise_weight0.01_2 &
nohup srun --gres=gpu:1 --mem=25GB --partition=titanx-long python train.py --train_file=data/stsa.binary.phrases.train --dev_file=data/stsa.binary.dev --test_file=data/stsa.binary.test --train_emb=True --word_keep=0.9 --noise_weight=0.01 --random_type=Adversarial > log.sst2.Adversarial.noise_weight0.01_3 &
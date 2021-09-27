python main.py --task train --method NASTAR \
--target_noise target_data/ACVacuum_7/pseudo.wav \
--eval_noise target_data/ACVacuum_7/train.wav \
--cohort_list target_data/ACVacuum_7/cohort.txt \
--device 2 --metric --eval_init
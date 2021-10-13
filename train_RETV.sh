for NOISE_TYPE in ACVacuum Babble CafeRestaurant Car MetroSubway; do
    python main.py --task train --method RETV \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --cohort_list target_data/${NOISE_TYPE}/cohort.txt \
    --use_source_noise
done
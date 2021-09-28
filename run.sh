for NOISE_TYPE in CafeRestaurant_7 Car_7 MetroSubway_7; do
    python main.py --task train --method GT \
    --target_noise target_data/${NOISE_TYPE}/train.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 0
done

for NOISE_TYPE in ACVacuum_7 Babble_7 CafeRestaurant_7 Car_7 MetroSubway_7; do
    python main.py --task train --method NASTAR_A05_K250 \
    --target_noise target_data/${NOISE_TYPE}/pseudo.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --cohort_list target_data/${NOISE_TYPE}/cohort.txt \
    --device 0 --use_source_noise
done
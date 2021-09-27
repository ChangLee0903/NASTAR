for NOISE_TYPE in ACVacuum_7 Babble_7 CafeRestaurant_7 Car_7 MetroSubway_7; do
    python main.py --task train --method RETV \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --cohort_list target_data/${NOISE_TYPE}/cohort.txt \
    --device 7
done

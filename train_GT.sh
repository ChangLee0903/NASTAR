for NOISE_TYPE in ACVacuum Babble CafeRestaurant Car MetroSubway; do
    python main.py --task train --method GT \
    --target_noise target_data/${NOISE_TYPE}/train.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav
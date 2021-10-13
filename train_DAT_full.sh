for NOISE_TYPE in ACVacuum Babble CafeRestaurant Car MetroSubway; do
    python main.py --task train --method DAT_full \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 2
done
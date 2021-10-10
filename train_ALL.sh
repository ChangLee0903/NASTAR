for NOISE_TYPE in ACVacuum Babble CafeRestaurant Car MetroSubway; do
    python main.py --task train --method ALL \
    --target_noise target_data/${NOISE_TYPE}/pseudo.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --use_source_noise --alpha 0.9
done
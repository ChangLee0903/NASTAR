for NOISE_TYPE in CafeRestaurant_7 Car_7 MetroSubway_7; do
    python main.py --task train --method ALL_A05 \
    --target_noise target_data/${NOISE_TYPE}/pseudo.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 3 --use_source_noise
done
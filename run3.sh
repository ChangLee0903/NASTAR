for NOISE_TYPE in CafeRestaurant_7 Car_7 MetroSubway_7; do
    python main.py --task train --method TEST \
    --target_noise target_data/${NOISE_TYPE}/test.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav \
    --device 2
done

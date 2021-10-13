for NOISE_TYPE in ACVacuum Babble CafeRestaurant Car MetroSubway; do
    python main.py --task train --method OPT \
    --target_noise target_data/${NOISE_TYPE}/test.wav \
    --eval_noise target_data/${NOISE_TYPE}/train.wav
done
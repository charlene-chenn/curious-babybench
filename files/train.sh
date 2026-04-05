for alpha in 0.5 0.75 1.0; do
    for seed in 1 2 3; do
        python train.py --alpha $alpha --seed $seed --episodes 500
    done
done
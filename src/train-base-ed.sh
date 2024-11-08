for l in 1e5 2e5 5e6
do
    python trainers/train-base-ed.py \
    --dataset-name ACE \
    --data-dir ../data/processed/ \
    --output-dir ../checkpoints/ \
    --model-name mistralai/Mistral-7B-v0.1 \
    --max-length 256 \
    --include-nan \
    --batch-size 8 \
    --learning-rate $l \
    --num-epochs 15 \
    --device cuda:0
done
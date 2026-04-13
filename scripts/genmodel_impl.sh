#!/bin/bash
# genmodel_impl.sh
# Train VAE, GAN, and Diffusion models on CelebA, then run evaluation

EPOCHS=100
TRAIN_RATIO=0.01
BATCH=128
SAVE_EVERY=10
LR=0.0002

LOG_VAE="logs/vae_training.log"
LOG_GAN="logs/gan_training.log"
LOG_DIFF="logs/diffusion_training.log"
LOG_EVAL="logs/genmodel_eval.log"

mkdir -p logs

echo "Starting Generative Model Training..."
echo "  Epochs:      ${EPOCHS}"
echo "  Train ratio: ${TRAIN_RATIO}"
echo "  Save every:  ${SAVE_EVERY} epochs"

# ── VAE ───────────────────────────────────────────────────────────────────────
echo "Training VAE..."
nohup python3 scripts/genmodel_impl.py \
    -m VAE \
    -e ${EPOCHS} \
    -t ${TRAIN_RATIO} \
    -b ${BATCH} \
    -s ${SAVE_EVERY} \
    --lr ${LR} \
    > ${LOG_VAE} 2>&1
echo "VAE done. Check ${LOG_VAE}"

# ── GAN ───────────────────────────────────────────────────────────────────────
echo "Training GAN..."
nohup python3 scripts/genmodel_impl.py \
    -m GAN \
    -e ${EPOCHS} \
    -t ${TRAIN_RATIO} \
    -b ${BATCH} \
    -s ${SAVE_EVERY} \
    --lr ${LR} \
    > ${LOG_GAN} 2>&1
echo "GAN done. Check ${LOG_GAN}"

# ── Diffusion ─────────────────────────────────────────────────────────────────
echo "Training Diffusion..."
nohup python3 scripts/genmodel_impl.py \
    -m Diffusion \
    -e ${EPOCHS} \
    -t ${TRAIN_RATIO} \
    -b ${BATCH} \
    -s ${SAVE_EVERY} \
    --lr ${LR} \
    > ${LOG_DIFF} 2>&1
echo "Diffusion done. Check ${LOG_DIFF}"

# ── Evaluation ────────────────────────────────────────────────────────────────
echo "Running evaluation..."
nohup python3 scripts/genmodel_eval.py \
    > ${LOG_EVAL} 2>&1
echo "Evaluation done. Check ${LOG_EVAL}"

echo "All done. Outputs in scripts/output_genmodel/"
echo "Monitor logs with:"
echo "  tail -f ${LOG_VAE}"
echo "  tail -f ${LOG_GAN}"
echo "  tail -f ${LOG_DIFF}"
echo "  tail -f ${LOG_EVAL}"

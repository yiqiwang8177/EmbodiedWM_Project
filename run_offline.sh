SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="can" # Any task of the respective suite
NUM_EXP_TRAJS=50
SEED=0
ckpt_path="./../ckpt_dirs/can_50/DP_Pretrain_base_policy_latest.pt"
conda activate ${SUITE}_env
usemppi=False
lps=True
HYDRA_FULL_ERROR=1 python3 train_offline.py \
    --configs cfg_dp_mppi ${SUITE}\
    --wandb_project EmbodiedWM_${SUITE} \
    --wandb_exp_name "seed${SEED}_mppi${usemppi}_lps${lps}" \
    --use_wandb False \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS} \
    --seed ${SEED} \
    --pretrained_base_ckpt ${ckpt_path} \
    --use_mppi ${usemppi} \
    --lps_reward ${lps}
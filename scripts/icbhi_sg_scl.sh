
MODEL="ast"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="sg_scl_bs8_lr5e-5_ep50_seed${s}_best_param"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method ce \
                                        --domain_adaptation2 \
                                        --alpha 1.0 \
                                        --proj_dim 768 \
                                        --temperature 0.06 \
                                        --meta_mode dev \
                                        --target_type project1_project2block \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done

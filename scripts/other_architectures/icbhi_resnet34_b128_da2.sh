
MODEL="resnet34"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="da2_bs128_lr5e-5_ep400_seed${s}_new"
        CUDA_VISIBLE_DEVICES=2 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 400 \
                                        --batch_size 128 \
                                        --optimizer adam \
                                        --learning_rate 1e-3 \
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
                                        --meta_mode dev \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done

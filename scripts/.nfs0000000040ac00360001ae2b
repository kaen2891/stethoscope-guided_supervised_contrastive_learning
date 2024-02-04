MODEL="ast"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}"
        CUDA_VISIBLE_DEVICES=0 python tsne.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --weighted_loss \
                                        --cosine \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --domain_adaptation \
                                        --meta_mode dev \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --from_sl_official \
                                        --domain_adaptation \
                                        --audioset_pretrained \
                                        --method ce \
                                        --print_freq 100 \
                                        --eval \
                                        --name da2_normal_for_paper_seed${s} \
                                        --pretrained \
                                        --pretrained_ckpt add_your_directory(checkpoint)                                        
                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        #--pretrained_ckpt ./save/icbhi_ast_patchmix_cl_bs8_lr5e-5_ep50_seed${s}/best.pth
                                        # --pretrained_ckpt /home/junewoo/stethoscope-guided_supervised_contrastive_learning/save/da2/icbhi_ast_ce_dev_sg_scl_bs8_lr5e-5_ep50_seed${s}_best_param/best.pth
    done
done

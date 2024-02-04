# Stethoscope-guided Supervised Contrastive Learning (ICASSP 2024)
[arXiv](https://arxiv.org/abs/2312.09603) | [BibTeX](#bibtex)


<p align="center">
<img width="704" alt="image" src="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/assets/46586785/d6f51658-df38-4a8b-8174-0cca9c8127fa">
</p>

Official Implementation of **Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification.**<br/>

**See you in ICASSP 2024!**

[June-Woo Kim](https://kaen2891.github.io/profile/),
[Sangmin Bae](https://www.raymin0223.com),
[Won-Yang Cho](https://github.com/wonyangcho),
[Byungjo Lee](https://github.com/bzlee-bio),
[Ho-Young Jung](https://scholar.google.com/citations?user=gvaE8RUAAAAJ&hl=en)$^\dagger$ <br/>
$^\dagger$ corresponding author


- We demonstrated that **addressing the domain inconsistency challenges by introducing domain adversarial training techniques.**
- We introduced **a novel stethoscope-guided supervised contrastive learning (SG-SCL) approach for cross-domain adaptation** with various pretrained architectures.
- The proposed method forces the model to **reduce the distribution shift between different stethoscope classes while maintaining equivalence in the same class**.


## Requirements
Install the necessary packages with: 
```
$ pip install torch torchvision torchaudio
$ pip install -r requirements.txt
```
For the reproducibility, we used `torch=2.0.7` and `torchaudio=2.0.`

## Data Preparation
Download the ICBHI dataset files from [official_page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).     
```bash
$ wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```
All `*.wav` and `*.txt` should be saved in `data/icbhi_dataset/audio_test_data`.     

Note that ICBHI dataset consists of a total of 6,898 respiratory cycles, 
of which 1,864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

<p align="center">
<img width="349" alt="image" src="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/assets/46586785/c971762d-9f7d-4f44-8b75-d49e90c30ddd">
</p>


## Training 
To simply train the model, run the shell files in `scripts/`.    
1. **`scripts/icbhi_ce.sh`**: Cross-Entropy loss with AST model.
2. **`scripts/icbhi_dat_device.sh`**: Cross-Entropy loss with Domain Adaptation (DANN) in terms of Device (stethoscope) with AST Model. 
3. **`scripts/icbhi_sg_scl.sh`**: Cross-Entropy loss with SG-SCL (Stethoscope-guided Supervised Contrastive Learning) with AST model.

Important arguments for different data settings.
- `--dataset`: other lungsound datasets or heart sound can be implemented.
- `--class_split`: "lungsound" or "diagnosis" classification.
- `--n_cls`: set number of classes as 4 or 2 (normal / abnormal) for lungsound classification.
- `--test_fold`: "official" denotes 60/40% train/test split, and "0"~"4" denote 80/20% split.
- **`--domain_adaptation`**: Using the proposed `DAT` in this paper.
- **`--domain_adaptation2`**: Using the proposed `SCL` in this paper.
- **`--meta_mode`**: meta information for cross-domain; choices=`['none', 'age', 'sex', 'loc', 'dev', 'label']`. The default is `dev`.

Important arguments for models.
- `--model`: network architecture, see [models](models/).
- `--from_sl_official`: load ImageNet pretrained checkpoint.
- `--audioset_pretrained`: load AudioSet pretrained checkpoint and only support AST and SSAST.

Important argument for evaluation.
- `--eval`: switch mode to evaluation without any training.
- `--pretrained`: load pretrained checkpoint and require `pretrained_ckpt` argument.
- `--pretrained_ckpt`: path for the pretrained checkpoint.

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/best.pth`.     

## Result

The proposed Stethoscope-Guided Supervised Contrastive Learning achieves a 61.71% Score, which is a significant improvement of 2.16% over the baseline.
<p align="center">
<img width="696" alt="image" src="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/assets/46586785/0cdeca6c-89b0-4cba-ab9f-4307f54c2c8d">
</p>

## T-SNE
To get the t-sne results, run the shell files in `scripts`.
1. **`scripts/get_tsne.sh`**: get the t-sne from pretrained weights. You must type the pretrained weight as below:
- `--eval`
- `--pretrained`
- **`--pretrained_ckpt`**: load pretraind weights. Please type your own model weight (e.g., `/home/junewoo/stethoscope-guided_supervised_contrastive_learning/save/da2/icbhi_ast_ce_dev_sg_scl_bs8_lr5e-5_ep50_seed${s}_best_param/best.pth`).

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{kim2023stethoscope,
  title={Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification},
  author={Kim, June-Woo and Bae, Sangmin and Cho, Won-Yang and Lee, Byungjo and Jung, Ho-Young},
  journal={arXiv preprint arXiv:2312.09603},
  year={2023}
}
```

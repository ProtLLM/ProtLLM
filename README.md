
# <img src="assets/logo.png" style="width: 5%"> ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training
[Le Zhuo*](https://le-zhuo.com/), Zewen Chi*, [Minghao Xu*](https://chrisallenming.github.io/), Heyan Huang, Heqi Zheng, [Conghui He](https://conghui.github.io/), Xian-Ling Mao, [Wentao Zhang](https://zwt233.github.io/)


<a href='https://protllm.github.io/project/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/pdf/2403.07920.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 

<div align=center><img src="assets/demo.gif" width="90%" height="90%" /></div>

This repository hosts the code, data and model weights of [**ProtLLM**](https://arxiv.org/abs/2403.07920), a versatile cross-modal large language model for both protein-centric and protein-language tasks.






## TODOs
- [ ] Release the code for retrieval.
- [ ] Release the raw InterPT dataset.
- [ ] Update the huggingface version of ProtLLM.
- [ ] ...


## Setup
### Enviroment
1. Clone this repository and navigate to the ProtLLM folder
```bash
git clone https://github.com/ProtLLM/ProtLLM.git
cd ProtLLM
```
2. Install Package
```Shell
conda create -n protllm python=3.10 -y
conda activate protllm
pip install e .
```

### Data & Checkpoints
We release the pre-processed version of our InterPT dataset, all datasets for downstream tasks, and pre-trained checkpoints in [Hugging Face](https://huggingface.co/datasets/ProtLLM/ProtLLM).
  

## Training

### Pre-training
For pre-training, you should download the pre-preprocessed dataset from [Hugging Face](https://huggingface.co/datasets/ProtLLM/ProtLLM) first and run the following script:
```Shell
bash scripts/pretrain.sh
```

### Fine-tuning
We provide the fine-tuning scripts to reproduce all results of ProtLLM on various protein-centric tasks, including Enzyme Commission (EC) number prediction, Gene Ontology (GO) term prediction, and Protein-Protein Interaction (PPI) prediction.
By default, we use the pre-trained ProtST-ESM-2 as the protein encoder, which can be downloaded from the [ProtST](https://github.com/DeepGraphLearning/ProtST) repository. After downloading the processed dataset from [Hugging Face](https://huggingface.co/datasets/ProtLLM/ProtLLM), you can run the following script to finetune ProtLLM on specific downstream task:
```Shell
bash scripts/finetune.sh
```
The detailed hyperparameters and settings for each task can be found in the appendix of our paper. Note that, we also fine-tune the weights of protein encoder for GO and EC prediction tasks, which can be done by setting `--lr_ratio` to 0.1 in the fine-tuning script.

## Evaluation

### Fine-tuning
After fine-tuning ProtLLM on protein-centric tasks, you can evaluate its performance by running the following script:
```Shell
bash scripts/eval.sh
```
Remember to set `--task` to the target task name and `--n_labels` to the number of labels of the task. You should also change the LoRA hyperparameters `--sft_lora_r` and `--sft_lora_alpha` to the values you used in the fine-tuning script. 

### In-context Learning
Run the following script to perform in-context learning with ProtLLM (using PPI prediction as an example):
```Shell
bash scripts/icl.sh
```
You can specify the `--n_demo` argument to control the number of demonstration samples.

## Contact
If you have any questions related to the code or the paper, feel free to contact [Le Zhuo](zhuole1025@gmail.com), [Zewen Chi](czw@bit.edu.cn), and [Minghao Xu](chrisallenming@gmail.com).


## Citation
If you find our work useful in your research, please consider citing ProtLLM:
```bibtex
@article{zhuo2024protllm,
  title={ProtLLM: An Interleaved Protein-Language LLM with Protein-as-Word Pre-Training},
  author={Le Zhuo and Zewen Chi and Minghao Xu and Heyan Huang and Heqi Zheng and Conghui He and Xian-Ling Mao and Wentao Zhang},
  journal={arXiv preprint arXiv:2403.079205},
  year={2024}
}
```

## From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery

Associated repository for "[From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery](https://arxiv.org/abs/2309.05203)" (AAAI 2024)

### Huggingface model checkpoints

We have uploaded our pre-trained checkpoints to [huggingface](https://huggingface.co/SCIR-HI).

Pre-trained checkpoints include:

+ [ada-t5-small](https://huggingface.co/SCIR-HI/ada-t5-small) (~77 million parameters)
+ [ada-t5-base](https://huggingface.co/SCIR-HI/ada-t5-base) (~250 million parameters)

### Example usage of pre-trained checkpoints

You can easily load these checkpoints using transformers:

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ada-t5-small
model = T5ForConditionalGeneration.from_pretrained("SCIR-HI/ada-t5-small")
tokenizer = AutoTokenizer.from_pretrained("SCIR-HI/ada-t5-small", model_max_length=512)

# ada-t5-base
model = T5ForConditionalGeneration.from_pretrained("SCIR-HI/ada-t5-base")
tokenizer = AutoTokenizer.from_pretrained("SCIR-HI/ada-t5-base", model_max_length=512)
```

### Datasets

We incorporated two new datasets: PseudoMD-1M, the first artificially real dataset for cross-modal molecule discovery, and DrugBank-23, a novel data source compared to current datasets.

For PseudoMD-1M, we have open-sourced it in [huggingface](https://huggingface.co/datasets/SCIR-HI/PseudoMD-1M).

For DrugBank-23, we are currently reviewing the licencse of raw data from [DrugBank](https://go.drugbank.com/), to ensure compliance with their policies. We will release it later.

### Environments
Our key dependencies are as follows:
| Package | Version |
| ---- | ---- |
| torch | 2.0.1 |
| transformers | 4.31.0 |
| python | 3.11.4 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |
| rdkit | 2023.3.2 |

We also provide the yml file of our conda environment. You can rebuild the environment using 
```
conda env create -f environment.yml
```

### Pre-training

Our code, built on PyTorch and transformers, is simple and customizable, involving no complex frameworks or trainers.
First, navigate to the 'pretrain' directory by running the command 
```
cd pretrain
```

Then, the simplest launching command is 
```
python run.py
```
or
```
./run.sh
```
The latter command will launch the training exactly the same way we train ada-t5. As shown in below:
| Parameters | N |
| ---- | ----|
| Training Steps | 100,000|
| Learning Rate | 1e-3|
| Batch Size | 128 |
| Warm-up Steps | 1000|
| Weight Decay| 0.1|
| Data Type | bf16 |
| Random Seed | 42 |

If you want to customize hyperparameters, just append them to `python run.py` or modify `run.sh`. You can refer to `myparser.py`, which contains all hyperparameters we use and has been well organized into groups.

### Fine-tuning
First, navigate to the 'finetune' directory by running the command 
```
cd finetune
```
Similar to pre-training, the simplest launching command is 
```
python run.py
```
or
```
./run.sh
```
The latter command will launch the training exactly the same way we use. As shown in below:
| Parameters | N |
| ---- | ----|
| Training Steps | 50,000|
| Learning Rate | 1e-3|
| Batch Size | 32 |
| Warm-up Steps | 1000|
| Weight Decay| 0.1|
| Data Type | bf16 |
| Random Seed | 42 |

If you want to customize hyperparameters, just append them to `python run.py` or modify `run.sh`. You can refer to `myparser.py`, which contains all hyperparameters we use and has been well organized into groups.

For testing, execute `./test.sh`, ensuring the path to the fine-tuned model weight is specified within `test.sh` before running.

### Generation settings

During validation and test, we use `T5ForConditionalGeneration.generate()` function to generate outputs. For generation config, we **only** specify `num_beams=1` and `max_new_tokens=512`. All other parameters keeps empty and depends on the default value of `generate()` function in [transformers(version 4.31.0)](https://huggingface.co/docs/transformers/v4.31.0/en/index).

### Citation
If you found our work useful, please cite:
```bibtex
@article{chen2023artificially,
  title={From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery},
  author={Chen, Yuhan and Xi, Nuwa and Du, Yanrui and Wang, Haochun and Jianyu, Chen and Zhao, Sendong and Qin, Bing},
  journal={arXiv preprint arXiv:2309.05203},
  year={2023}
}
```
[Tiếng Việt](https://github.com/Kokoroou/self-supervised-segmentation/blob/main/README_vi.md)

# Self-supervised learning for medical image segmentation

## Introduction
In the medical field, it is quite difficult to collect a large amount of high-quality data to train machine learning 
models. On the one hand, medical data is often sensitive, related to the personal information of patients. On the other
hand, labeling medical data also requires the intervention of medical experts.

In this thesis, a self-supervised learning method for the medical image segmentation model will be tested. The success
of the thesis will open up a new approach for training machine learning models in the medical field, without relying on
too much data.

## Dataset
The dataset used in this thesis:

- [PolypGen2021](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312)

## Model architecture
- Masked Autoencoder
- UNETR

The model tested in the thesis is built based on the following paper:
[Self Pre-training with Masked Autoencoders for Medical Image Classification and 
Segmentation](https://arxiv.org/abs/2203.05573)

![Model Architecture](https://raw.githubusercontent.com/Kokoroou/self-supervised-segmentation/main/services/ui/image/implemented_model_architecture.png)

## Requirements
- Python 3.9

## Usage

Clone this repository to your environment:
```bash
git clone https://github.com/Kokoroou/self-supervised-segmentation.git
cd self-supervised-segmentation
```

### Research

#### Environment setup
```bash
pip install -r research/requirements.txt
```
#### Train model
```bash
# Train Masked Autoencoder model
python research/main_autoencoder.py train -m mae_vit_base_patch16 -s {path_to_dataset}

# Train Segmentation model
python research/main_segmentation.py train -m vit_mae_seg_base -s {path_to_train_dataset} -t {path_to_test_dataset}
```

#### Use model
```bash
# Use Masked Autoencoder model
python research/main_autoencoder.py infer -m mae_vit_base_patch16 -c {path_to_checkpoint} -i {path_to_image}

# Use Segmentation model
python research/main_segmentation.py infer -m vit_mae_seg_base -c {path_to_checkpoint} -i {path_to_image}
```

#### Help
```bash
# Help for Masked Autoencoder model
python research/main_autoencoder.py -h
python research/main_autoencoder.py train -h
python research/main_autoencoder.py test -h
python research/main_autoencoder.py infer -h

# Help for Segmentation model
python research/main_segmentation.py -h
python research/main_segmentation.py train -h
python research/main_segmentation.py test -h
python research/main_segmentation.py infer -h
``` 

### Application

#### Environment setup
```bash
pip install -r services/requirements.txt
pip install -r services/requirements_pytorch.txt
pip install -e ./services
```

#### Run application
```bash
python services/main_streamlit.py
```

### Implement with Docker
```bash
docker build -t self-supervised-segmentation .
docker container run -d --name vit_streamlit -p 8585:8585 -m 3g --cpus 2 vit_streamlit
```

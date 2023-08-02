# Học tự giám sát cho mô hình phân vùng ảnh y khoa

## Giới thiệu
Trong lĩnh vực y tế, việc thu thập được nhiều dữ liệu chất lượng để huấn luyện các mô hình học máy là tương đối 
khó khăn. Một mặt, dữ liệu y tế thường là những dữ liệu nhạy cảm, có liên quan đến thông tin cá nhân của người bệnh. 
Mặt khác, việc gán nhãn cho dữ liệu y tế cũng cần đến sự can thiệp của các chuyên gia y tế. 

Trong đồ án này, một phương pháp học tự giám sát cho mô hình phân vùng ảnh y khoa sẽ được thử nghiệm. Thành công của
đồ án sẽ mở ra một hướng tiếp cận mới cho việc huấn luyện các mô hình học máy trong lĩnh vực y tế, mà không cần phụ 
thuộc vào lượng dữ liệu quá nhiều.

## Dữ liệu
Bộ dữ liệu được sử dụng trong đồ án này:

- [PolypGen2021](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312)

## Kiến trúc mô hình
- Masked Autoencoder
- UNETR

Mô hình thử nghiệm trong đồ án được xây dựng dựa trên bài báo sau:
[Self Pre-training with Masked Autoencoders for Medical Image Classification and 
Segmentation](https://arxiv.org/abs/2203.05573)

![Kiến trúc mô hình](https://raw.githubusercontent.com/Kokoroou/self-supervised-segmentation/main/services/ui/image/implemented_model_architecture.png)

## Yêu cầu
- Python 3.9

## Sử dụng

Clone repository này về môi trường của bạn:
```bash
git clone https://github.com/Kokoroou/self-supervised-segmentation.git
cd self-supervised-segmentation
```

### Nghiên cứu

#### Cài đặt môi trường
```bash
pip install -r research/requirements.txt
```
#### Huấn luyện mô hình
```bash
# Huấn luyện mô hình Masked Autoencoder
python research/main_autoencoder.py train -m mae_vit_base_patch16 -s {path_to_dataset}

# Huấn luyện mô hình Segmentation
python research/main_segmentation.py train -m vit_mae_seg_base -s {path_to_train_dataset} -t {path_to_test_dataset}
```

#### Sử dụng mô hình
```bash
# Sử dụng mô hình Masked Autoencoder
python research/main_autoencoder.py infer -m mae_vit_base_patch16 -c {path_to_checkpoint} -i {path_to_image}

# Sử dụng mô hình Segmentation
python research/main_segmentation.py infer -m vit_mae_seg_base -c {path_to_checkpoint} -i {path_to_image}
```

#### Trợ giúp
```bash
# Trợ giúp cho mô hình Masked Autoencoder
python research/main_autoencoder.py -h
python research/main_autoencoder.py train -h
python research/main_autoencoder.py test -h
python research/main_autoencoder.py infer -h

# Trợ giúp cho mô hình Segmentation
python research/main_segmentation.py -h
python research/main_segmentation.py train -h
python research/main_segmentation.py test -h
python research/main_segmentation.py infer -h
``` 

### Ứng dụng

#### Cài đặt môi trường
```bash
pip install -r services/requirements.txt
pip install -r services/requirements_pytorch.txt
pip install -e ./services
```

#### Chạy ứng dụng
```bash
python services/main_streamlit.py
```

### Triển khai với Docker
```bash
docker build -t self-supervised-segmentation .
docker container run -d --name vit_streamlit -p 8585:8585 -m 3g --cpus 2 vit_streamlit
```

# MoGCN
## What is it?
MoGCN, a multi-omics integration method based on graph convolutional network.<br>
![Image text](https://github.com/Lifoof/MoGCN/blob/master/data/Figs1.png)
As shown in figure, inputs to the model are multi-omics expression matrices, including but not limited to genomics, transcriptomics, proteomics, etc. MoGCN exploits the GCN model to incorporate and extend two unsupervised multi-omics integration algorithms: Autoencoder algorithm (AE) based on expression matrix and similarity network fusion algorithm based on patient similarity network. Feature extraction is not necessary before AE and SNF. <br>

## Requirements 
MoGCN is a Python scirpt tool, Python environment need:<br>
Python 3.6 or above <br>
Pytorch 1.4.0 or above <br>
snfpy 0.2.2 <br>


## Usage
The whole workflow is divided into three steps: <br>
* Use AE to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix <br>
* Use SNF to construct patient similarity network <br>
* Input multi-omics feature matrix  and the patient similarity network to GCN <br>
The sample data is in the data folder, which contains the CNV, mRNA and RPPA data of BRCA. <br>
### Command Line Tool
```Python
python AE_run.py -p1 data/fpkm_data.csv -p2 data/gistic_data.csv -p3 data/rppa_data.csv -m 0 -s 0 -d cpu
python SNF.py -p data/fpkm_data.csv data/gistic_data.csv data/rppa_data.csv -m sqeuclidean
python GCN_run.py -fd result/latent_data.csv -ad result/SNF_fused_matrix.csv -ld data/sample_classes.csv -ts data/test_sample.csv -m 1 -d gpu -p 20
```
The meaning of the parameters can be viewed through -h/--help <br>

### Data Format
* The input type of each omics data must be .csv, the rows represent samples, and the columns represent features (genes). In each expression matrix, the first column must be the samples, and the remaining columns are features. Samples in all omics data must be consistent. AE and SNF are unsupervised models and do not require sample labels.<br>
* GCN is a semi-supervised classification model, it requires sample label files (.csv format) during training. The first column of the label file is the sample name, the second column is the digitized sample label, the remaining columns are not necessary. <br>

## Contact
For any questions please contact Dr. Xiao Li (Email: lixiaoBioinfo@163.com).

## License
MIT License

## Citation
Li X, Ma J, Leng L, Han M, Li M, He F and Zhu Y (2022) MoGCN: A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis. Front. Genet. 13:806842. doi: 10.3389/fgene.2022.806842. <br>








clinical data:   https://www.linkedomics.org/data_download/TCGA-BRCA/?utm_source=chatgpt.com
https://linkedomics.org/data_download/TCGA-BRCA/Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi


Tôi đã thực hiện một số cải tiến quan trọng để tăng khả năng biểu hiện của mô hình GCN:
Thêm Residual Connections (Kết nối phần dư):
Giúp mô hình học tốt hơn với các lớp sâu hơn
Giảm vấn đề vanishing gradient
Cho phép thông tin từ các lớp trước được truyền trực tiếp đến các lớp sau
Thêm Batch Normalization:
Chuẩn hóa dữ liệu giữa các lớp
Giúp quá trình huấn luyện ổn định hơn
Tăng tốc độ hội tụ
Thêm Attention Mechanism:
Tạo lớp AttentionLayer mới
Cho phép mô hình tập trung vào các đặc trưng quan trọng
Tăng khả năng học các mối quan hệ phức tạp
Tăng độ sâu của mô hình:
Thêm một lớp GCN thứ ba
Tăng khả năng học các đặc trưng phức tạp hơn
Cải thiện Dropout:
Đặt giá trị mặc định cho dropout là 0.5
Thêm dropout cho lớp GCN mới
Giúp giảm overfitting
Cấu trúc lại forward pass:
Tổ chức rõ ràng hơn với các comment
Thêm residual connections ở mỗi lớp
Áp dụng attention sau mỗi lớp GCN
Những cải tiến này sẽ giúp mô hình:
Học được các đặc trưng phức tạp hơn
Ổn định hơn trong quá trình huấn luyện
Giảm overfitting
Tăng khả năng biểu diễn tổng thể
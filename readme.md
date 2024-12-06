<h1 align="center">Deep Learning for Mpox Skin Lesion Classification</h1>
<p align="center">Habibah Ratna Fadhila Islami Hana</p>
<p align="center">2311110038</p>

![dataset-cover](https://github.com/user-attachments/assets/469e6711-a3d0-4b53-8df9-0ba45f96f53c)

## Introduction
Penyakit monkeypox (Mpox) semakin menjadi perhatian karena penyebarannya yang terus meningkat di berbagai negara. Salah satu gejala utama penyakit ini adalah munculnya lesi pada kulit yang sering kali mirip dengan gejala penyakit kulit lain, seperti cacar air atau smallpox. Kondisi ini membuat diagnosis menjadi tantangan besar, terutama karena kesalahan diagnosis dapat menyebabkan penundaan penanganan dan memperburuk penyebaran.

Kemajuan teknologi, khususnya di bidang deep learning, membuka peluang baru untuk membantu mengenali lesi kulit secara otomatis. Dengan memanfaatkan dataset citra medis, seperti Mpox Skin Lesion Dataset (MSLD v2.0), model deep learning dapat dilatih untuk mengenali pola dan karakteristik spesifik dari lesi Mpox. Model ini mampu memberikan hasil yang cepat dan akurat, sehingga dapat mendukung tenaga medis dalam membuat keputusan diagnostik.

Pemanfaatan dataset Mpox Skin Lesion bertujuan untuk mengeksplorasi penerapan deep learning dalam klasifikasi lesi kulit Mpox. Dengan menggunakan teknik-teknik terbaru dalam pengolahan citra medis, diharapkan hasilnya dapat menghasilkan model yang akurat dan efisien. Pengembangan model ini diharapkan mampu mendukung pembuatan alat diagnostik otomatis yang dapat membantu tenaga medis dalam mendeteksi Mpox lebih cepat dan lebih akurat, sehingga memungkinkan deteksi dini dan penanganan yang lebih efektif terhadap penyebaran penyakit. Dataset ini dapat diakses melalui link berikut: [Source](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data)

## Data
Dataset ini terbagi menjadi dua folder utama:
#### 1. Original Images
Folder ini berisi gambar-gambar asli yang digunakan dalam proses pelatihan, pengujian, dan validasi model. Di dalam folder ini terdapat subfolder bernama `FOLDS` yang berisi lima fold (fold1 hingga fold5) untuk melakukan cross-validation 5-fold. Setiap fold memiliki folder terpisah untuk `test`, `train`, dan `validation` set, yang memungkinkan pengujian model dengan data yang berbeda-beda untuk mengukur kinerjanya secara lebih akurat. Masing-masing berisikan citra dari kulit dengan 6 kelas berbeda.

#### 2. Augmented Images
Folder ini berisi gambar yang telah melalui proses data augmentation untuk meningkatkan jumlah dan keragaman data dalam rangka mendukung tugas klasifikasi. Gambar-gambar yang telah diaugmentasi disimpan dalam subfolder bernama `FOLDS_AUG`, yang berisi gambar-gambar dari masing-masing fold yang ada di folder "FOLDS" pada Original Images. Augmented images terdiri dari 5 folds (lipatan) yang di dalamnya hanya terdapat folder Train untuk masing-masing kelas. 

## Commands
Untuk melatih model, dapat digunakan perintah berikut:
```python
python train.py
```

Untuk melakukan evaluasi, dapat digunakan perintah berikut:
```python
python test.py
```

## Model
![image](https://github.com/user-attachments/assets/cf6e732c-fb21-4e64-a466-f87bdcd7c48d)

Model yang digunakan di sini adalah Convolutional Neural Network (CNN) sederhana yang dirancang untuk mengolah gambar dan mengklasifikasikannya ke dalam salah satu dari enam kategori. Input gambarnya berukuran 32 x 32 piksel dengan tiga saluran warna (RGB). Arsitektur model ini terdiri dari dua blok utama. Setiap blok punya lapisan Convolution, ReLU Activation, dan MaxPooling.

Blok pertama menggunakan 20 filter dengan ukuran kernel 5 x 5 untuk mengekstrak fitur awal dari gambar. Setelah melewati fungsi aktivasi ReLU yang menambahkan elemen non-linearitas, data dilanjutkan ke lapisan MaxPooling berukuran 2 x 2, yang bertugas mengurangi dimensi spasial gambar. Lalu, blok kedua bekerja dengan cara serupa, tetapi kali ini menggunakan 50 filter dengan ukuran kernel yang sama. Setelah fitur lebih kompleks diekstrak, lapisan ReLU Activation dan MaxPooling kembali digunakan untuk menyaring data lebih lanjut.

Setelah kedua blok tersebut, output berupa data multidimensi akan dilakukan flattening, yaitu diubah menjadi vektor satu dimensi (1D). Vektor ini lalu melewati lapisan Dropout dengan probabilitas 50% untuk mengurangi risiko overfitting. Di tahap akhir, vektor ini diproses oleh lapisan fully connected (Dense) layer, yang akan mengubahnya menjadi output berukuran 6, sesuai dengan jumlah kategori dalam dataset. 

## Result
### 1. Cross Entropy Loss Function
![SimpleCNN](https://github.com/user-attachments/assets/9aaf2ebf-ec84-468b-8547-cc78f7431bbc)

Grafik tersebut menunjukkan perubahan Cross Entropy Loss selama proses pelatihan (training) dan validasi (validation) model simple CNN pada data Mpox skin lesion. Secara umum, baik training loss maupun validation loss menunjukkan pola penurunan seiring bertambahnya epoch, yang menandakan bahwa model sedang belajar dari data dan semakin mampu memprediksi dengan baik. Terdapat sedikit kesenjangan antara training loss dan validation loss, di mana training loss lebih rendah. Namun, kesenjangan ini tidak terlalu besar, sehingga model tidak tampak mengalami overfitting yang signifikan. Pada akhir epoch (sekitar epoch 8 hingga 10), baik training loss maupun validation loss telah mencapai titik konvergensi yang menandakan bahwa model telah mencapai performa optimal dengan dataset yang ada. Cross Entropy Loss yang rendah pada akhir pelatihan menunjukkan bahwa model memiliki kemampuan prediktif yang baik terhadap data Mpox skin lesion. 

### 2. Accuracy, Precision, Recall, dan F1 Score
![image](https://github.com/user-attachments/assets/e695493f-0984-4a89-87a3-38376bc7000c)

- Accuracy (0.99450): Model berhasil membuat prediksi yang benar sebesar `99,45%` dari keseluruhan data. Tingginya nilai accuracy ini menunjukkan bahwa model memiliki performa yang sangat baik secara keseluruhan.
- Precision (0.99459): Nilai ini mengindikasikan bahwa dari semua prediksi positif yang dibuat oleh model, `99,459%` adalah benar-benar positif (true positive). Hal ini berarti model jarang memberikan false positive, sehingga dapat dipercaya dalam memprediksi kelas positif.
- Recall (0.99450): Nilai ini menunjukkan bahwa dari semua kasus positif yang ada, model mampu mendeteksi `99,45%` dengan benar. Hal ini berarti model memiliki sensitivitas yang sangat baik, yang penting untuk menghindari false negative pada kasus klinis seperti deteksi lesi kulit Mpox.
- F1 Score (0.99451): Sebagai rata-rata harmonik dari Precision dan Recall, nilai F1 Score sebesar `99,451%` menunjukkan bahwa model memiliki keseimbangan yang sangat baik antara kemampuan mendeteksi kelas positif (recall) dan menghasilkan prediksi positif yang benar (precision).

### 3. Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/86122326-340b-4cbb-af92-de8e3f28c7cf)

- Chickenpox (Baris 1):
    - Model berhasil mengklasifikasikan 98% gambar sebagai Chickenpox dengan benar.
    - Sebanyak 2,4% gambar yang seharusnya Chickenpox salah diklasifikasikan sebagai HFMD.
- Cowpox (Baris 2):
    - Semua gambar Cowpox diklasifikasikan dengan benar (100% akurasi untuk kelas ini).
- Healthy (Baris 3):
    - Model mengklasifikasikan 100% gambar yang termasuk kategori Healthy dengan benar tanpa kesalahan.
- HFMD (Baris 4):
    - Model berhasil mengklasifikasikan 98% gambar sebagai HFMD dengan benar.
    - Sebanyak 2% gambar yang seharusnya HFMD salah diklasifikasikan sebagai Chickenpox.
- Measles (Baris 5):
    - Model mengklasifikasikan 100% gambar sebagai Measles dengan benar.
- Monkeypox (Baris 6):
    - Model mengklasifikasikan 100% gambar sebagai Monkeypox dengan benar tanpa kesalahan.

### 4. ROC-AUC
![roc_auc](https://github.com/user-attachments/assets/58537900-d471-4d63-8f1f-37e9eb284c81)

Grafik ROC dan nilai AUC menunjukkan bahwa model deep learning bekerja dengan sangat baik. Hampir semua kategori punya nilai AUC mendekati 1.0, yang berarti modelnya mampu membedakan setiap kelas dengan sangat akurat. Untuk Chickenpox, Cowpox, Healthy, HFMD, dan Measles, nilai AUC-nya sempurna di 1.0, sedangkan Monkeypox sedikit lebih rendah di 0.99, tapi itu tetap nilai yang sangat tinggi. Hal ini berarti, model hampir selalu benar dalam mengidentifikasi Monkeypox, walaupun ada sedikit kemungkinan kesalahan kecil. 

Kalau dilihat dari kurva ROC-nya, semua kategori mendekati sudut kiri atas, menunjukkan tingkat sensitivitas tinggi dengan false positive yang sangat rendah. Dibandingkan dengan model acak (AUC = 0.50), model ini jelas jauh lebih unggul. Dari segi penggunaan, model ini sangat cocok untuk diagnosis lesi kulit seperti Chickenpox, Cowpox, kulit sehat, HFMD, Measles, dan Monkeypox. Tapi, untuk Monkeypox, meskipun hasilnya hampir sempurna, tetap ada baiknya waspada pada kemungkinan kesalahan kecil yang bisa terjadi.
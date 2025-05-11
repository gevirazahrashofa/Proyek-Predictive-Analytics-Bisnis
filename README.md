# Laporan Proyek Predictive Analytics Bisnis - House Rent Prediction
**Nama: Gevira Zahra Shofa**

## Domain Proyek

Ketersediaan hunian yang layak dan terjangkau menjadi aspek penting dalam pembangunan berkelanjutan, terutama di wilayah dengan urbanisasi tinggi seperti aglomerasi Yogyakarta. Pertumbuhan penduduk mendorong meningkatnya permintaan rumah sewa dan fluktuasi harga yang dipengaruhi oleh berbagai faktor, seperti karakteristik rumah tangga, fisik bangunan, dan lingkungan sekitar.

Di era modern, pasar penyewaan rumah memainkan peran vital dalam mendukung mobilitas penduduk dan menjadi sumber pendapatan bagi pemilik properti. Penentuan harga sewa yang akurat sangat penting bagi pemilik, penyewa, maupun agen properti untuk memastikan keadilan, efisiensi, dan daya saing di pasar.

### Latar Belakang

Secara tradisional, penentuan harga sewa rumah seringkali dilakukan secara subjektif berdasarkan intuisi pemilik atau perbandingan sederhana terhadap properti serupa di sekitar lokasi. Pendekatan ini memiliki sejumlah keterbatasan:
+ Rentan terhadap bias pribadi yang dapat menyebabkan harga tidak adil atau tidak sesuai pasar.
+ Proses penilaian manual memakan waktu dan tidak skalabel di pasar yang dinamis.
+ Sulit menangkap kompleksitas hubungan antara faktor-faktor seperti lokasi, luas bangunan, atau aksesibilitas terhadap harga sewa.

Prediksi harga sewa rumah yang akurat sangat penting untuk menciptakan pasar yang adil dan efisien. Bagi pemilik, hal ini membantu menetapkan harga yang tepat agar properti cepat tersewa tanpa merugi. Bagi penyewa, prediksi harga membantu dalam merencanakan anggaran dan menghindari harga yang tidak wajar. Selain itu, prediksi yang akurat juga mendukung pengambilan keputusan bagi agen properti, pengembang, dan investor dalam menentukan strategi bisnis mereka.

Untuk menjawab tantangan ini, pendekatan berbasis machine learning menawarkan solusi yang andal. Dengan memanfaatkan data historis, machine learning dapat mengungkap pola kompleks dan membangun model prediktif yang:
+ Memberikan estimasi harga sewa dengan tingkat akurasi tinggi.
+ Mengidentifikasi faktor-faktor dominan yang memengaruhi harga.
+ Mengadaptasi model terhadap dinamika pasar secara real time.
+ Menghasilkan output secara cepat dan efisien dalam skala besar.

Referensi: [Analisis Faktor-Faktor yang Mempengaruhi Harga Rumah di Area Aglomerasi Yogyakarta](https://ejournal.undip.ac.id/index.php/pwk/article/view/37603)

## Business Understanding

### Problem Statements

+ Bagaimana membangun model machine learning yang akurat dan andal untuk memprediksi harga sewa rumah berdasarkan fitur-fitur seperti ukuran, jumlah kamar, jumlah kamar mandi, jenis area, kota, status perabotan, dan preferensi penyewa?
+ Fitur properti mana yang paling memengaruhi harga sewa rumah, dan bagaimana tingkat pengaruh relatif masing-masing fitur terhadap variasi harga?
+ Bagaimana memilih dan mengoptimalkan model terbaik untuk mencapai prediksi harga yang akurat, efisien, dan mampu beradaptasi dengan data baru?

### Goals

+ Mengembangkan model prediksi harga sewa rumah dengan kesalahan rendah (MSE, RMSE, MAE) dan akurasi tinggi (R-squared) menggunakan data historis properti.
+ Mengidentifikasi fitur paling berpengaruh terhadap harga sewa dengan:
  + Analisis korelasi.
  + Interpretasi feature importance dari model seperti Random Forest dan Gradient Boosting.
+ Membandingkan dan mengoptimalkan berbagai model machine learning (Linear Regression, Random Forest, Gradient Boosting) untuk mencari kombinasi terbaik melalui evaluasi dan tuning hyperparameter.

### Solution Statements

+ Mengembangkan tiga model regresi yang berbeda, yaitu Regresi Linear, Random Forest Regressor, dan Gradient Boosting Regressor, untuk memprediksi harga sewa rumah. Semua model akan dievaluasi dengan metrik seperti MSE, RMSE, MAE, dan R-squared. Model-model ini dipilih karena:
  + Regresi Linear: Sebagai model dasar untuk memahami hubungan linear antara fitur dan target.
  + Random Forest Regressor: Mampu menangani non-linearitas dan memberikan informasi tentang pentingnya fitur.
  + Gradient Boosting Regressor: Model ensemble yang kuat dengan potensi akurasi tinggi.
+ Melakukan hyperparameter tuning pada model Random Forest Regressor menggunakan teknik GridSearchCV. Tujuannya adalah meningkatkan performa model dengan kombinasi parameter optimal. Hyperparameter yang akan di tuning:
  + n_estimators: Jumlah pohon dalam hutan.
  + max_depth: Kedalaman maksimum pohon.
  + min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node internal.
  + min_samples_leaf: Jumlah minimum sampel yang diperlukan untuk berada di node daun.
+ Membandingkan performa sebelum dan sesudah tuning untuk memastikan peningkatan akurasi, dan memilih model akhir yang paling optimal untuk digunakan dalam prediksi nyata.

## Data Understanding

Pada tahap ini, dilakukan eksplorasi dan pemahaman mendalam tentang dataset yang akan digunakan untuk membangun model prediksi harga sewa rumah. Tujuannya adalah untuk mendapatkan wawasan tentang karakteristik data, mengidentifikasi potensi masalah, dan membuat keputusan yang tepat dalam tahap data preparation selanjutnya.

### Informasi tentang dataset

+ Sumber Dataset: Dataset yang digunakan adalah "House Rent Prediction Dataset" yang dapat diakses melalui platform Kaggle.
+ Tautan Dataset: [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset).
+ Dataset memiliki 4746 sample dengan 12 fitur.
+ Dataset memiliki 4 fitur bertipe int64 dan 8 fitur bertipe object.

### Variabel-variabel pada House Rent Prediction dataset adalah sebagai berikut:

+ Rent: Harga sewa rumah (dalam satuan mata uang lokal India, Rupee). Ini adalah variabel target yang akan diprediksi oleh model. Tipe data: Numerik.
+ Size: Ukuran rumah dalam satuan square feet. Tipe data: Numerik.
+ BHK: Jumlah kamar tidur, hall, dan dapur dalam properti. Tipe data: Numerik.
+ Bathroom: Jumlah kamar mandi dalam properti. Tipe data: Numerik.
+ Area Type: Tipe area properti (misalnya, 'Carpet Area', 'Built Area', 'Super Area'). Tipe data: Kategorikal.
+ City: Kota tempat properti berada. Tipe data: Kategorikal.
+ Furnishing Status: Status perabotan properti (misalnya, 'Furnished', 'Unfurnished', 'Semi-Furnished'). Tipe data: Kategorikal.
+ Tenant Preferred: Preferensi jenis penyewa yang disukai (misalnya, 'Family', 'Bachelor', 'Any'). Tipe data: Kategorikal.
+ Floor: Lantai tempat properti berada. Tipe data: Kategorikal.
+ Area Locality: Lokasi spesifik dalam kota. Tipe data: Kategorikal.
+ Posted On: Tanggal iklan properti diposting. Tipe data: Objek (Tanggal).
+ Point of Contact: Cara menghubungi pemilik properti. Tipe data: Kategorikal.

### Pemeriksaan Missing Values

Pemeriksaan dilakukan untuk mengetahui kolom yang memiliki nilai hilang agar dapat ditangani dengan tepat.

<img src="https://drive.google.com/uc?export=view&id=1fPQ7YwrQ_1Z6g8N-_dcxNvLlyBZQEyug" width="200"/>

Tidak ada missing values pada dataset, sehingga tidak diperlukan penanganan khusus.

### Univariate Analysis

Analisis ini mengevaluasi distribusi kategori dan angka pada setiap fitur untuk memahami pola umum data serta mendeteksi outlier atau ketidakseimbangan.

#### Distribusi Variabel Kategorikal

<img src="https://drive.google.com/uc?export=view&id=1wGzE8CBMQ2cDjrTlFE-xDHT_YbmV738u" width="250"/>

Hanya terdapat 2 sample Built Area pada fitur Area Type, maka kedua sample ini akan dihapus.

<img src="https://drive.google.com/uc?export=view&id=179V9qGfw7AmLN8scUgRSlbQQrzu-cZ5Z" width="300"/>

Fitur Floor dan Area Locality memiliki banyak sekali nilai unik, maka kedua fitur ini juga akan dihapus.

<img src="https://drive.google.com/uc?export=view&id=1apBlYz7wvdIOI3N44D9bfXivG_qhoIa5" width="300"/>

Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.
+ City: Properti paling banyak berasal dari Mumbai, paling sedikit dari Kolkata.
+ Furnishing Status: Mayoritas properti berstatus Semi-Furnished.
+ Tenant Preferred: Mayoritas properti menerima Bachelors/Family.

#### Distribusi Variabel Numerik

<img src="https://drive.google.com/uc?export=view&id=1SeVqYjR2D-ReG2duLlYppQgJcQ4g4OT3" width="300"/>

Distribusi Rent dan Size condong ke kanan (right-skewed), menunjukkan banyak properti dengan harga sewa dan ukuran rendah. Sebagian kecil data menunjukkan nilai ekstrem (outlier).

### Multivariate Analysis

Multivariate analysis dilakukan untuk memahami hubungan antar beberapa variabel dalam dataset yang telah dibersihkan.

#### Analisis untuk Fitur Numerik

+ Menambahkan Fitur Price_per_sqft

GAMBAR

Menambahkan kolom baru dengan rumus (Rent * 1000) / Size untuk menghitung harga sewa per sqft, agar perbandingan properti lebih objektif dan memudahkan deteksi outlier.

+ Menghapus Outlier Ukuran per BHK

GAMBAR

Menghapus data dengan rasio Size/BHK < 300 karena ukuran terlalu kecil per kamar tidur dianggap tidak realistis dan berpotensi merupakan kesalahan input.

+ Menghapus Outlier Price_per_sqft

GAMBAR

Menghapus data di luar satu standar deviasi dari rata-rata Price_per_sqft per kota untuk menghindari bias geografis dan menjaga data tetap relevan.

+ Menghapus Outlier Jumlah Kamar Mandi

GAMBAR

Menghapus data dengan jumlah kamar mandi lebih dari BHK + 2 karena dianggap tidak wajar dan kemungkinan merupakan kesalahan input.

+ Menghapus Kolom Price_per_sqft

GAMBAR

Menghapus kolom setelah digunakan untuk filtering karena tidak dibutuhkan dalam model akhir, agar dataset lebih ringkas.

+ Melihat kolerasi antara semua fitur numerik

GAMBAR

#### Analisis untuk Fitur Kategorik

+ Rata-rata 'Rent' Relatif terhadap Area Type

GAMBAR

+ Rata-rata 'Rent' Relatif terhadap City

GAMBAR

+ Rata-rata 'Rent' Relatif terhadap Furnishing Status

GAMBAR

+ Rata-rata 'Rent' Relatif terhadap Tenand Preferred

GAMBAR

## Data Preparation

Pada tahap ini, dilakukan pembersihan, transformasi, dan pemilihan data untuk memastikan kualitas dan relevansi data yang digunakan. 

+ One-Hot Encoding

Kolom kategorikal seperti *Area Type*, *City*, *Furnishing Status*, dan *Tenant Preferred* diubah menggunakan teknik One-Hot Encoding karena model machine learning hanya dapat memproses data numerik. Metode ini dipilih karena tidak memberikan urutan pada kategori, dan penggunaan `drop_first=True` membantu menghindari multicollinearity antar variabel dummy.

GAMBAR

+ Pembagian Data (Train-Test Split)

Pembagian data dilakukan dengan teknik Train-Test Split, menggunakan rasio 80% untuk training dan 20% untuk testing. Pembagian ini penting untuk melatih model pada data training dan menguji kinerjanya pada data testing untuk mengukur kemampuan generalisasi. random_state=42 digunakan agar pembagian data tetap konsisten setiap kali eksekusi ulang.

GAMBAR

+ Normalisasi Fitur Numerik

Fitur numerik seperti Size, BHK, dan Bathroom dinormalisasi menggunakan teknik Min-Max Scaling. Ini dilakukan karena fitur-fitur tersebut memiliki skala yang berbeda, yang dapat memengaruhi kinerja model berbasis jarak atau gradien. MinMaxScaler mengubah nilai ke dalam rentang 0-1 agar setiap fitur memberikan kontribusi yang seimbang. Normalisasi dilakukan dengan scaler.fit_transform() pada data training dan scaler.transform() pada data testing untuk mencegah kebocoran data.

GAMBAR

## Modelling

Dalam tahap ini, beberapa algoritma regresi diimplementasikan, dievaluasi, dan dibandingkan untuk menemukan model terbaik yang sesuai dengan tujuan proyek.

### Algoritma yang Digunakan
#### Regresi Linear:
#### Random Forest Regressor:
#### Gradient Boosting Regressor:
### Hyperparameter Tuning
### Pemilihan Model Terbaik
### Kesimpulan Modeling

## Evaluation

# Laporan Proyek Predictive Analytics Bisnis - House Rent Prediction
**Nama: Gevira Zahra Shofa**

## Domain Proyek

Pasar persewaan rumah di kota-kota besar India seperti Bangalore, Mumbai, dan Delhi mengalami pertumbuhan yang pesat seiring dengan meningkatnya urbanisasi, mobilitas penduduk, dan dinamika ekonomi. Dalam konteks ini, penetapan harga sewa yang akurat dan adil menjadi aspek krusial bagi pemilik properti, penyewa, dan agen real estat.

Penentuan harga sewa tidak hanya dipengaruhi oleh lokasi, tetapi juga oleh berbagai karakteristik properti seperti ukuran, jumlah kamar tidur, jumlah kamar mandi, status perabotan, dan jenis area. Memahami hubungan antara fitur-fitur ini dan harga sewa menjadi kunci untuk menciptakan pasar yang lebih efisien dan transparan.

### Latar Belakang

Secara tradisional, penetapan harga sewa rumah seringkali dilakukan secara subjektif, berdasarkan intuisi pemilik atau perbandingan informal dengan properti sejenis. Pendekatan ini memiliki sejumlah kelemahan:

+ Rentan terhadap bias dan ketidakakuratan harga.
+ Tidak efisien dalam skala besar, terutama di pasar yang cepat berubah.
+ Tidak mempertimbangkan pengaruh multivariat dari berbagai fitur secara bersamaan.

Dengan memanfaatkan data historis dari berbagai kota besar di India, pendekatan berbasis machine learning dapat mengidentifikasi pola-pola kompleks dalam data dan menghasilkan prediksi harga sewa yang lebih akurat. Pendekatan ini bermanfaat bagi:

+ Pemilik properti: Menentukan harga optimal agar properti cepat tersewa tanpa merugi.
+ Penyewa: Membantu menyusun anggaran dan mengenali harga pasar yang wajar.
+ Agen properti dan pengembang: Menyusun strategi bisnis berbasis data.

Referensi: [Analisis Faktor-Faktor yang Mempengaruhi Harga Rumah di Area Aglomerasi Yogyakarta](https://ejournal.undip.ac.id/index.php/pwk/article/view/37603)

## Business Understanding

### Problem Statements

+ Bagaimana membangun model machine learning untuk memprediksi harga sewa berdasarkan fitur seperti ukuran, kamar, kamar mandi, perabotan, jenis area, dan kota?
+ Fitur mana yang paling berpengaruh terhadap harga sewa, dan seberapa besar pengaruhnya?
+ Bagaimana memilih dan mengoptimalkan model terbaik agar prediksi harga sewa lebih akurat dan efisien?

### Goals

+ Membangun model prediksi harga sewa rumah dengan error rendah (MSE, RMSE, MAE) dan akurasi tinggi (R²) menggunakan data dari berbagai kota besar di India.
+ Mengidentifikasi fitur paling berpengaruh terhadap harga sewa melalui analisis korelasi dan interpretasi feature importance dari model.
+ Membandingkan dan mengoptimalkan berbagai algoritma machine learning untuk memilih model terbaik.

### Solution Statements

+ Membangun tiga model regresi utama:
  + Linear Regression: Sebagai baseline untuk hubungan linear antara fitur dan target.
  + Random Forest Regressor: Untuk menangani hubungan non-linear dan mengevaluasi pentingnya fitur.
  + Gradient Boosting Regressor: Model ensemble dengan potensi akurasi tinggi dalam prediksi.
+ Melakukan hyperparameter tuning pada model Random Forest Regressor menggunakan teknik GridSearchCV. Tujuannya adalah meningkatkan performa model dengan kombinasi parameter optimal. Hyperparameter yang akan di tuning:
  + `n_estimators`: Jumlah pohon dalam hutan.
  + `max_depth`: Kedalaman maksimum pohon.
  + `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi node internal.
  + `min_samples_leaf`: Jumlah minimum sampel yang diperlukan untuk berada di node daun.
+ Membandingkan performa model sebelum dan sesudah tuning, serta memilih model akhir yang paling optimal untuk prediksi harga sewa nyata.

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

![image](https://github.com/user-attachments/assets/e3ea36db-d78c-4824-a2af-428522724cf1)

Tidak ada missing values pada dataset, sehingga tidak diperlukan penanganan khusus.

### Univariate Analysis

Analisis ini mengevaluasi distribusi kategori dan angka pada setiap fitur untuk memahami pola umum data serta mendeteksi outlier atau ketidakseimbangan.

#### Distribusi Variabel Kategorikal
![image](https://github.com/user-attachments/assets/c63f47d5-ced0-443b-ad67-c4e71612b577)

+ Hanya terdapat 2 sample Built Area pada fitur Area Type, maka kedua sample ini akan dihapus.

![image](https://github.com/user-attachments/assets/58e0516d-e638-4b8c-b392-8f47126cbc77)

+ Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.
  + City: Properti paling banyak berasal dari Mumbai, paling sedikit dari Kolkata.
  + Furnishing Status: Mayoritas properti berstatus Semi-Furnished.
  + Tenant Preferred: Mayoritas properti menerima Bachelors/Family.

![image](https://github.com/user-attachments/assets/d7888680-fa75-42ce-a4a9-f8c924120505)

+ Fitur Floor dan Area Locality memiliki banyak sekali nilai unik, maka kedua fitur ini juga akan dihapus.

#### Distribusi Variabel Numerik

![image](https://github.com/user-attachments/assets/edfccfd5-a9da-4193-9944-093ba4ea5907)

+ Sebagian besar properti memiliki 1 hingga 3 kamar tidur, dengan 2 BHK sebagai yang paling umum.
+ Distribusi sewa miring ke kanan, menunjukkan sebagian besar properti disewa dengan harga rendah hingga sedang, namun ada outlier dengan harga sangat tinggi.
+ Sebagian besar properti berukuran kecil hingga menengah; ukuran besar (di atas 3000 sqft) sangat jarang.
+ Properti dengan 1 atau 2 kamar mandi mendominasi, sejalan dengan jumlah kamar tidur.

### Multivariate Analysis

Multivariate analysis dilakukan untuk memahami hubungan antar beberapa variabel dalam dataset yang telah dibersihkan.

#### Analisis untuk Fitur Numerik

+ Menambahkan Fitur Price_per_sqft

![image](https://github.com/user-attachments/assets/49cfe0d0-6cb1-414f-bedb-749df2cd4cbb)

Menambahkan kolom baru dengan rumus (Rent * 1000) / Size untuk menghitung harga sewa per sqft, agar perbandingan properti lebih objektif dan memudahkan deteksi outlier.

+ Menghapus Outlier Ukuran per BHK
  
Menghapus data dengan rasio Size/BHK < 300 karena ukuran terlalu kecil per kamar tidur dianggap tidak realistis dan berpotensi merupakan kesalahan input.

![image](https://github.com/user-attachments/assets/59e9c1c5-29f6-492b-a5c2-67979517ef5b)

Dataset awal memiliki 4744 baris. Setelah menghapus outlier berdasarkan kriteria luas per BHK < 300 sqft, tersisa 4196 baris. Artinya, 548 data dianggap outlier dan dihapus untuk meningkatkan kualitas analisis dan akurasi model prediksi.

+ Menghapus Outlier Price_per_sqft

Menghapus data di luar satu standar deviasi dari rata-rata Price_per_sqft per kota untuk menghindari bias geografis dan menjaga data tetap relevan.

![image](https://github.com/user-attachments/assets/6dd4bef5-dfa3-4880-9664-d9ecad889d72)

Setelah menghapus outlier berdasarkan 'Price per sqft', jumlah data berkurang dari 4196 menjadi 3699 baris. Ini menunjukkan ada 497 data yang ekstrem dalam hal harga per sqft dan dihapus untuk menjaga validitas model.

+ Menghapus Outlier Jumlah Kamar Mandi

Menghapus data dengan jumlah kamar mandi lebih dari BHK + 2 karena dianggap tidak wajar dan kemungkinan merupakan kesalahan input.

![image](https://github.com/user-attachments/assets/632a36e4-dd4f-49c7-806e-c000c1976590)

Outlier berdasarkan jumlah kamar mandi dihapus, data hanya berkurang 3 baris (dari 3699 ke 3696). Artinya, outlier jenis ini relatif sedikit.

+ Menghapus Kolom Price_per_sqft

Menghapus kolom setelah digunakan untuk filtering karena tidak dibutuhkan dalam model akhir, agar dataset lebih ringkas.

+ Melihat kolerasi antara semua fitur numerik
  
![image](https://github.com/user-attachments/assets/dc138c11-330b-4817-be27-f409718d331d)

  + BHK & Bathroom: Korelasi sangat tinggi (1).
  + Size & (BHK, Bathroom): Korelasi positif kuat (0.79, 0.75).
  + Rent & (BHK, Size, Bathroom): Korelasi positif sedang (0.5, 0.5, 0.6).

#### Analisis untuk Fitur Kategorik

+ Rata-rata 'Rent' Relatif terhadap Area Type
  
![image](https://github.com/user-attachments/assets/77437e38-f35a-4d6f-aa5c-ed1171a29e01)

Carpet Area" memiliki harga sewa rata-rata yang jauh lebih tinggi daripada "Super Area

+ Rata-rata 'Rent' Relatif terhadap City
  
![image](https://github.com/user-attachments/assets/3062b126-fd6b-4647-950e-dcdbec551de9)

Mumbai memiliki rata-rata harga sewa jauh lebih tinggi dari kota lain.

+ Rata-rata 'Rent' Relatif terhadap Furnishing Status
  
![image](https://github.com/user-attachments/assets/3607058b-c6ef-4ea3-a6c6-1129572ed7f2)

Properti Furnished memiliki rata-rata sewa tertinggi, diikuti Semi-Furnished, lalu Unfurnished, menunjukkan bahwa semakin lengkap perabot, semakin tinggi harga sewanya.

+ Rata-rata 'Rent' Relatif terhadap Tenand Preferred

![image](https://github.com/user-attachments/assets/64ef768a-abc2-462e-9c8f-b95f9e4b2e40)

Properti yang disewakan untuk Family memiliki rata-rata sewa tertinggi, disusul Bachelors Tenand Preferred, lalu Bachelors/Family, menunjukkan preferensi penyewa memengaruhi besaran harga sewa.

## Data Preparation

Pada tahap ini, dilakukan pembersihan, transformasi, dan pemilihan data untuk memastikan kualitas dan relevansi data yang digunakan. 

+ One-Hot Encoding

Kolom kategorikal seperti *Area Type*, *City*, *Furnishing Status*, dan *Tenant Preferred* diubah menggunakan teknik One-Hot Encoding karena model machine learning hanya dapat memproses data numerik. Metode ini dipilih karena tidak memberikan urutan pada kategori, dan penggunaan `drop_first=True` membantu menghindari multicollinearity antar variabel dummy.

![image](https://github.com/user-attachments/assets/9f76b796-33bb-4684-8057-706955e44d45)

+ Pembagian Data (Train-Test Split)

Pembagian data dilakukan dengan teknik Train-Test Split, data dibagi menjadi 80% untuk training (2956 data) dan 20% untuk testing (740 data). Ini memastikan model bisa belajar dari data yang cukup dan diuji pada data yang belum pernah dilihat.

![image](https://github.com/user-attachments/assets/cceaca77-f99a-4b93-b88e-2fa5e9426b8d)

+ Normalisasi Fitur Numerik

Fitur numerik seperti Size, BHK, dan Bathroom dinormalisasi menggunakan teknik Min-Max Scaling. Ini dilakukan karena fitur-fitur tersebut memiliki skala yang berbeda, yang dapat memengaruhi kinerja model berbasis jarak atau gradien. MinMaxScaler mengubah nilai ke dalam rentang 0-1 agar setiap fitur memberikan kontribusi yang seimbang. Normalisasi dilakukan dengan scaler.fit_transform() pada data training dan scaler.transform() pada data testing untuk mencegah kebocoran data.

![image](https://github.com/user-attachments/assets/e454092f-9b67-4f5a-bd7a-0f7c7b693a44)

## Modelling

Dalam tahap ini, beberapa algoritma regresi diimplementasikan, dievaluasi, dan dibandingkan untuk menemukan model terbaik yang sesuai dengan tujuan proyek.

### Algoritma yang Digunakan

#### Regresi Linear:

Regresi Linear adalah model sederhana yang memetakan hubungan antara fitur dan target dalam bentuk garis lurus. Model ini cepat, mudah diinterpretasikan, dan cocok jika hubungan antar variabel bersifat linear. Namun, model ini kurang efektif jika terdapat hubungan non-linear atau outlier yang signifikan, karena sifatnya yang sensitif terhadap data ekstrem dan asumsi linearitas. Tahapan dan parameter nya:

+ Model: `LinearRegression()` dari `sklearn.linear_model`
+ Pelatihan: `fit(X_train, y_train)`
+ Tidak memiliki hyperparameter utama yang perlu diatur

#### Random Forest Regressor:

Random Forest adalah model ensemble yang menggabungkan banyak pohon keputusan, menghasilkan prediksi yang stabil dan akurat untuk data kompleks. Model ini dapat menangani non-linearitas dan outlier dengan baik serta cenderung tidak overfitting. Kelemahannya terletak pada interpretasi yang lebih sulit dan waktu komputasi yang lebih besar dibanding regresi linear. Tahapan dan parameter nya:

+ Model: `RandomForestRegressor()` dari `sklearn.ensemble`
+ Pelatihan: `fit(X_train, y_train)`
+ Hyperparameter:
  + `n_estimators` (jumlah pohon)
  + `random_state` (reproduksibilitas hasil)

#### Gradient Boosting Regressor:

Gradient Boosting membangun pohon secara bertahap untuk memperbaiki kesalahan model sebelumnya, menghasilkan model yang sangat akurat dan mampu menangani hubungan non-linear. Meskipun performanya tinggi, model ini rentan overfitting jika tidak dituning dengan baik dan membutuhkan waktu pelatihan yang lebih lama. Tahapan dan Parameter:

+ Model: GradientBoostingRegressor() dari sklearn.ensemble
+ Pelatihan: fit(X_train, y_train)
+ Hyperparameter:
  + random_state (reproduksibilitas hasil)

### Hyperparameter Tuning

Tuning dilakukan pada model Random Forest Regressor untuk meningkatkan akurasi dan mencegah overfitting. Proses ini menggunakan `GridSearchCV` dari `sklearn.model_selection`, yang mengevaluasi kombinasi parameter terbaik melalui cross-validation `(cv=5)`. Parameter yang Dituning:

+ n_estimators: [50, 100, 200]
+ max_depth: [10, 20, None]
+ min_samples_split: [2, 5, 10]
+ min_samples_leaf: [1, 2, 4]

## Evaluation

Evaluasi model dilakukan untuk menilai seberapa baik model dalam memprediksi harga sewa rumah. Dalam proyek ini, digunakan empat metrik regresi utama yang memberikan gambaran menyeluruh terhadap akurasi dan kesalahan prediksi model:

### Mean Squared Error (MSE)

MSE menghitung rata-rata dari kuadrat selisih antara nilai aktual (yᵢ) dan nilai prediksi (ŷᵢ). Karena selisihnya dikuadratkan, MSE memberikan penalti besar terhadap kesalahan prediksi yang besar.

`MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`

Alasan Penggunaan:
MSE sangat berguna untuk mengidentifikasi model yang membuat kesalahan besar, karena ia memberikan bobot lebih pada outlier. Cocok untuk mendeteksi apakah model membuat prediksi yang benar-benar meleset.

### Root Mean Squared Error (RMSE)

RMSE adalah akar dari MSE. Nilai ini mengembalikan satuan ke dalam unit target (dalam proyek ini: Rupee), sehingga lebih mudah diinterpretasikan secara langsung sebagai rata-rata error prediksi.

`RMSE = √MSE`

Alasan Penggunaan:
RMSE memberikan gambaran yang intuitif tentang seberapa jauh, secara rata-rata, prediksi menyimpang dari nilai asli. Sangat berguna dalam konteks bisnis karena menggunakan satuan yang sama dengan harga sewa.

### Mean Absolute Error (MAE)

MAE menghitung rata-rata dari selisih absolut antara prediksi dan nilai aktual. Berbeda dengan MSE/RMSE, MAE tidak memperbesar efek dari outlier.

`MAE = (1/n) * Σ|yᵢ - ŷᵢ|`

Alasan Penggunaan:
MAE memberikan gambaran rata-rata error secara seimbang tanpa memperbesar kesalahan besar. Berguna sebagai pembanding MSE/RMSE untuk melihat apakah error ekstrem mendistorsi evaluasi model.

### R-squared (R²)

R² mengukur proporsi variasi pada variabel target (y) yang dapat dijelaskan oleh variabel input (X). Nilainya antara 0 dan 1, di mana semakin mendekati 1 berarti model menjelaskan lebih banyak variasi dalam data.

`R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)`

Alasan Penggunaan:
R² menunjukkan seberapa baik model menjelaskan data. Dalam konteks regresi harga, ini penting untuk menilai apakah model menangkap hubungan antar fitur dengan baik atau tidak.

Berikut adalah hasil evaluasi untuk setiap model yang dikembangkan:


| Model                       | MSE                 | RMSE                | MAE                 | R²                  |
| --------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Regresi Linear              | 297106384.62        | 17236.77            | 11325.07            | 0.71                |
| Random Forest Regressor     | 133767120.09        | 11565.77            | 6792.76             | 0.87                |
| Gradient Boosting Regressor | 117975329.97        | 10861.64            | 6600.55             | 0.88                |

Model Gradient Boosting Regressor menunjukkan kinerja terbaik secara keseluruhan. Ini ditunjukkan oleh nilai R² tertinggi sebesar 0.88, serta nilai MAE dan RMSE paling rendah dibandingkan model lainnya. Hal ini menunjukkan bahwa model ini paling akurat dalam memprediksi harga sewa rumah. Regresi Linear menunjukkan kinerja terburuk, yang mengindikasikan bahwa hubungan antara fitur dan harga sewa bersifat non-linear dan tidak dapat ditangkap oleh model linier sederhana. Sementara itu, Random Forest Regressor memberikan hasil yang baik dengan R² sebesar 0.87, walaupun sudah dilakukan Hyperparameter Tuning, namun hasilnya masih sedikit di bawah Gradient Boosting.

Kesimpulan keseluruhan : 

Model Gradient Boosting Regressor terbukti paling akurat dalam memprediksi harga sewa rumah, mengungguli Random Forest dan Linear Regression. Fitur seperti ukuran rumah dan jumlah kamar mandi berpengaruh besar terhadap harga. Model ini bermanfaat bagi agen properti atau penyewa untuk estimasi harga sewa secara cepat. Namun, model masih terbatas karena belum mempertimbangkan faktor eksternal seperti lokasi detail atau kondisi pasar. Ke depan, model dapat ditingkatkan dengan penambahan fitur, tuning hyperparameter, dan uji pada data lain. Hasil ini sejalan dengan penelitian lain yang juga menunjukkan keunggulan Gradient Boosting dalam prediksi harga properti.


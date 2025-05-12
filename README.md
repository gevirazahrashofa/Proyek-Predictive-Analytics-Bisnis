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
  + `n_estimators`: Jumlah pohon dalam hutan.
  + `max_depth`: Kedalaman maksimum pohon.
  + `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi node internal.
  + `min_samples_leaf`: Jumlah minimum sampel yang diperlukan untuk berada di node daun.
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

+ Sebagian besar properti memiliki 1 hingga 3 kamar tidur, dengan 2 BHK sebagai yang paling umum.
+ Distribusi sewa miring ke kanan, menunjukkan sebagian besar properti disewa dengan harga rendah hingga sedang, namun ada outlier dengan harga sangat tinggi.
+ Sebagian besar properti berukuran kecil hingga menengah; ukuran besar (di atas 3000 sqft) sangat jarang.
+ Properti dengan 1 atau 2 kamar mandi mendominasi, sejalan dengan jumlah kamar tidur.

### Multivariate Analysis

Multivariate analysis dilakukan untuk memahami hubungan antar beberapa variabel dalam dataset yang telah dibersihkan.

#### Analisis untuk Fitur Numerik

+ Menambahkan Fitur Price_per_sqft

Menambahkan kolom baru dengan rumus (Rent * 1000) / Size untuk menghitung harga sewa per sqft, agar perbandingan properti lebih objektif dan memudahkan deteksi outlier.

+ Menghapus Outlier Ukuran per BHK

<img src="https://drive.google.com/uc?export=view&id=1F69SYZtsliEENr-9UqTUfX_8Z-hjkQWo" width="300"/>

Menghapus data dengan rasio Size/BHK < 300 karena ukuran terlalu kecil per kamar tidur dianggap tidak realistis dan berpotensi merupakan kesalahan input.

+ Menghapus Outlier Price_per_sqft

<img src="https://drive.google.com/uc?export=view&id=1xa2b7wAbmsPR_vf0DAfKZtJ0odrJecz6" width="300"/>

Menghapus data di luar satu standar deviasi dari rata-rata Price_per_sqft per kota untuk menghindari bias geografis dan menjaga data tetap relevan.

+ Menghapus Outlier Jumlah Kamar Mandi

<img src="https://drive.google.com/uc?export=view&id=1w7AZiMx5yxPMz9eIynOCuXEtlKqBul87" width="300"/>

Menghapus data dengan jumlah kamar mandi lebih dari BHK + 2 karena dianggap tidak wajar dan kemungkinan merupakan kesalahan input.

+ Menghapus Kolom Price_per_sqft

Menghapus kolom setelah digunakan untuk filtering karena tidak dibutuhkan dalam model akhir, agar dataset lebih ringkas.

+ Melihat kolerasi antara semua fitur numerik

<img src="https://drive.google.com/uc?export=view&id=1DFRnNUlkvXdmMsm0j0oGeXrSOSMJxl1D" width="300"/>

+ BHK & Bathroom: Korelasi sangat tinggi (1).
+ Size & (BHK, Bathroom): Korelasi positif kuat (0.79, 0.75).
+ Rent & (BHK, Size, Bathroom): Korelasi positif sedang (0.5, 0.5, 0.6).

#### Analisis untuk Fitur Kategorik

+ Rata-rata 'Rent' Relatif terhadap Area Type

<img src="https://drive.google.com/uc?export=view&id=1dJeMBeW3zzTKkwd--48D0F6Y9lMPU-OE" width="500"/>


Carpet Area" memiliki harga sewa rata-rata yang jauh lebih tinggi daripada "Super Area

+ Rata-rata 'Rent' Relatif terhadap City

<img src="https://drive.google.com/uc?export=view&id=1CoAlMXPKq-kutJ3FSICo6jorDCu2U9RS" width="500"/>

Mumbai memiliki rata-rata harga sewa jauh lebih tinggi dari kota lain.

+ Rata-rata 'Rent' Relatif terhadap Furnishing Status

<img src="https://drive.google.com/uc?export=view&id=1PGcqHd3THPaxI4wQ3lWHppLimgBToczW" width="500"/>

Properti Furnished memiliki rata-rata sewa tertinggi, diikuti Semi-Furnished, lalu Unfurnished, menunjukkan bahwa semakin lengkap perabot, semakin tinggi harga sewanya.

+ Rata-rata 'Rent' Relatif terhadap Tenand Preferred

<img src="https://drive.google.com/uc?export=view&id=1xTa2O5p8EajrqEkN0_YzZg8I_tHlBnjR" width="500"/>

Properti yang disewakan untuk Family memiliki rata-rata sewa tertinggi, disusul Bachelors Tenand Preferred, lalu Bachelors/Family, menunjukkan preferensi penyewa memengaruhi besaran harga sewa.

## Data Preparation

Pada tahap ini, dilakukan pembersihan, transformasi, dan pemilihan data untuk memastikan kualitas dan relevansi data yang digunakan. 

+ One-Hot Encoding

Kolom kategorikal seperti *Area Type*, *City*, *Furnishing Status*, dan *Tenant Preferred* diubah menggunakan teknik One-Hot Encoding karena model machine learning hanya dapat memproses data numerik. Metode ini dipilih karena tidak memberikan urutan pada kategori, dan penggunaan `drop_first=True` membantu menghindari multicollinearity antar variabel dummy.

<img src="https://drive.google.com/uc?export=view&id=1xZ6S0AcYrcDkR_l4gGyfzeOcFRKWIqFL" width="400"/>

+ Pembagian Data (Train-Test Split)

Pembagian data dilakukan dengan teknik Train-Test Split, menggunakan rasio 80% untuk training dan 20% untuk testing. Pembagian ini penting untuk melatih model pada data training dan menguji kinerjanya pada data testing untuk mengukur kemampuan generalisasi. random_state=42 digunakan agar pembagian data tetap konsisten setiap kali eksekusi ulang.

<img src="https://drive.google.com/uc?export=view&id=1PCT-jSkWzefuGA8xa9BN516HqoWfUe8G" width="300"/>

Data dibagi menjadi 2956 sampel untuk pelatihan (dengan 13 fitur dan target) dan 740 sampel untuk pengujian (juga dengan 13 fitur dan target).

+ Normalisasi Fitur Numerik

Fitur numerik seperti Size, BHK, dan Bathroom dinormalisasi menggunakan teknik Min-Max Scaling. Ini dilakukan karena fitur-fitur tersebut memiliki skala yang berbeda, yang dapat memengaruhi kinerja model berbasis jarak atau gradien. MinMaxScaler mengubah nilai ke dalam rentang 0-1 agar setiap fitur memberikan kontribusi yang seimbang. Normalisasi dilakukan dengan scaler.fit_transform() pada data training dan scaler.transform() pada data testing untuk mencegah kebocoran data.

<img src="https://drive.google.com/uc?export=view&id=1wkh_X1N-ZTnFgBMozJM_iB4KBfmx8rQc" width="400"/>

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

### Pemilihan Model Terbaik

Metrik Evaluasi: R-squared (R²)

Model terbaik dipilih berdasarkan nilai R² tertinggi pada data testing. Setelah membandingkan performa Regresi Linear, Random Forest (sebelum dan sesudah tuning), dan Gradient Boosting, model dengan nilai R² tertinggi dipilih karena paling akurat dalam menjelaskan variabilitas target dan memiliki generalisasi terbaik terhadap data baru.

## Evaluation

Evaluasi model dilakukan untuk menilai seberapa baik model dalam memprediksi harga sewa rumah. Dalam proyek ini, digunakan empat metrik regresi utama yang memberikan gambaran menyeluruh terhadap akurasi dan kesalahan prediksi model:

### Mean Squared Error (MSE)

MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Metrik ini menekankan kesalahan besar karena selisih dikalikan dengan dirinya sendiri. Semakin kecil nilai MSE, semakin baik kinerja model.

`MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`

### Root Mean Squared Error (RMSE)

RMSE adalah akar dari MSE dan memiliki satuan yang sama dengan target (harga sewa), sehingga lebih mudah diinterpretasikan. Nilai RMSE yang rendah menunjukkan model melakukan prediksi dengan error yang kecil.

`RMSE = √MSE`

### Mean Absolute Error (MAE)

MAE menghitung rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya. Tidak seperti MSE/RMSE, MAE tidak memperbesar pengaruh outlier. Metrik ini memberikan gambaran seberapa besar kesalahan rata-rata tanpa arah.

`MAE = (1/n) * Σ|yᵢ - ŷᵢ|`

### R-squared (R²)

R² menunjukkan seberapa besar variasi harga sewa yang bisa dijelaskan oleh model. Nilai R² berada antara 0 dan 1, dengan nilai mendekati 1 menandakan bahwa model mampu menjelaskan hampir seluruh variasi data target.

`R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)`

Berikut adalah hasil evaluasi untuk setiap model yang dikembangkan:


| Model                       | MSE                 | RMSE                | MAE                 | R²                  |
| --------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| Regresi Linear              | 297106384.62        | 17236.77            | 11325.07            | 0.71                |
| Random Forest Regressor     | 133767120.09        | 11565.77            | 6792.76             | 0.87                |
| Gradient Boosting Regressor | 117975329.97        | 10861.64            | 6600.55             | 0.88                |

Model Gradient Boosting Regressor menunjukkan kinerja terbaik secara keseluruhan. Ini ditunjukkan oleh nilai R² tertinggi sebesar 0.88, serta nilai MAE dan RMSE paling rendah dibandingkan model lainnya. Hal ini menunjukkan bahwa model ini paling akurat dalam memprediksi harga sewa rumah. Regresi Linear menunjukkan kinerja terburuk, yang mengindikasikan bahwa hubungan antara fitur dan harga sewa bersifat non-linear dan tidak dapat ditangkap oleh model linier sederhana. Sementara itu, Random Forest Regressor memberikan hasil yang baik dengan R² sebesar 0.87, walaupun sudah dilakukan Hyperparameter Tuning, namun hasilnya masih sedikit di bawah Gradient Boosting.

Kesimpulan keseluruhan : 

Model Gradient Boosting Regressor terbukti paling akurat dalam memprediksi harga sewa rumah, mengungguli Random Forest dan Linear Regression. Fitur seperti ukuran rumah dan jumlah kamar mandi berpengaruh besar terhadap harga. Model ini bermanfaat bagi agen properti atau penyewa untuk estimasi harga sewa secara cepat. Namun, model masih terbatas karena belum mempertimbangkan faktor eksternal seperti lokasi detail atau kondisi pasar. Ke depan, model dapat ditingkatkan dengan penambahan fitur, tuning hyperparameter, dan uji pada data lain. Hasil ini sejalan dengan penelitian lain yang juga menunjukkan keunggulan Gradient Boosting dalam prediksi harga properti.

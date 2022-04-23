## Churn Prediction

**Project description** 

Dalam contoh kasus ini, akan dibuat model churn prediction untuk sebuah perusahaan telekomunikasi yang menjual layanan internet nirkabel dengan kartu prabayar. Tidak sedikit pelanggan perusahaan ini yang telah berpindah langganan ke perusahaan pesaing akibat tawaran harga dan layanan yang lebih menarik. Pihak manajemen perusahaan ini menyadari terkait masalah tersebut dan berencana meluncurkan program-program promosi untuk menahan churn rate. Program promosi ini hanya akan ditawarkan melalui SMS kepada kelompok pelanggan yang dianggap rawan churn. Agar lebih efektif, machine learning diperlukan untuk menentukan kelompok pelanggan tsb. 

### Code & Dataset

[Code](/https://colab.research.google.com/drive/113UrgppJhIu03hzfXRpOVVFYntkxO7eG?usp=sharing)

[dataset](/https://drive.google.com/file/d/1_eoIAiaWA53vCb5vepO52pxJSKp2anST/view?usp=sharing)

### 1. Input Data

```javascript
import pandas as pd
df1=pd.read_csv('churnprediction_ch9.csv', sep=',', index_col=['customer_id'])
df1.head(10)
df1.info()
```


### 2. Menampilkan Persentase Pelanggan Aktif dan Pelanggan Churn

```javascript
dfAktifChurn = df1.groupby('churn').count()
import matplotlib.pyplot as plt
plt.pie(dfAktifChurn['product'],labels=['Aktif', 'Churn'], autopct='%1.0f%%')
plt.title('Persentase Pelanggan Aktif vs Churn')
plt.axis('equal')
plt.show()
```
<img src="images/download.png?raw=true"/>

Terlihat bahwa di dalam dataset ada dua kelompok pelanggan, yaitu 20% pelanggan yang sudah churn dan sisanya 80% yang masih aktif.

### 3. Melihat Isi Kolom
```javascript
df1['product'].value_counts()
```
Karena masih berupa teks berisi nama-nama produk ("Kartu A", "Kartu B", "Kartu C"), maka feature ini harus diubah menjadi numerik. Cara yang paling umum adalah dengan metode one-hot encoding, yaitu menuliskan semua nulai yang mungkin muncul menjadi kolom kemudian memberikan nilai 0 dan 1 di kolom tersebut, tergantung pada pelanggan yang diwakili di setiap baris apakah menggunakan produk yang bersangkutan. Dalam kasus kita, karena ada tiga produk maka akan ada tiga kolom baru.

```javascript
pd.get_dummies(df1['product'])
```
Seperti yang kita harapkan, ada tiga kolom baru untuk ketiga nama produk. Pelanggan yang pertama karena menggunakan produk "Kartu A", maka nilainya 1 di kolom produk tersebut, dan lainnya bernilai 0. Demikian juga pelanggan yang paling bawah kita tahu menggunakan produk "Kartu C" karena kolom tersebut bernilai 1. 

Selanjutnya kita gabungkan ketiga kolom baru tersebut dengan df1, dan disimpan sebagai dataframe baru, yaitu df2. Karena sudah diwakili tiga kolom baru tadi, maka feature "product" bisa dibuang dari dataset dengan perintah drop():

```javascript
df2 = pd.concat([df1, pd.get_dummies(df1['product'])], axis=1, sort=False)
df2.drop(['product'], axis=1, inplace=True)
dfKorelasi = df2.corr()
```


### 4. Feature Selection

```javascript
import seaborn as sns
dfKorelasi = df2.corr()
sns.heatmap(dfKorelasi, xticklabels=dfKorelasi.columns.values, yticklabels=dfKorelasi.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(10,10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
```
<img src="images/download (1).png?raw=true"/>

Kita lihat dari heatmap di atas bahwa feature "reload_1" dan "reload_2" memiliki angka korelasi yang tinggi, yaitu 0.92 sehingga kita bisa memilih salah satunya saja untuk proses training. Demikian pula halnya dengan pasangan "socmed_1" dan "socmed_2" yang memiliki korelasi tinggi sehingga bisa kita buang salah satunya. Hal yang sama terjadi pada "music" dan "games". 

Proses feature selection di atas menghasilkan kesimpulan, yaitu kita akan membuang feature-feature ini dari proses training. "reload_2", "socmed_2", dan "games".

```javascript
X = df2.drop(['reload_2', 'socmed_2', 'games', 'churn'], axis=1, inplace=False)
y = df2['churn']
```

Dari dataset ini, kita ambil 80% sebagai training dataset dan sisanya 20% sebagai test dataset. Untuk keperluan ini kita pergunakan modul model_selection.

```javascript
import sklearn.model_selection as ms
X_train,X_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.8,random_state=0)
```
Setelah program diatas dijalankan, X_train akan berisi semua feature dan y_train berisi target yang akan kita pakai untuk proses training model. Sementara itu X_test dan y_test akan berisi test dataset yang akan kita pakai untuk mengukur kinerja model. 

Satu hal terakhir yang perlu dilakukan adalah normalisasi data atau disebut juga feature scalling, yaitu "memadatkan" semua feature agar seragam jangkauan nilai minimum dan maksimumnya. Ini penting agar proses pembuatan model bisa lebih lancar.

```javascript
import sklearn.preprocessing as pp
scl = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
```
### 5. Melatih Model

```javascript
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as met
model = lm.LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
```
Sampai titik ini, model telah terbentuk. Langkah selanjutnya kita ukur kinerja model ini dengan test dataset, kemudian tampiplkan nilai akurasi, precision, recall, dan AUC-nya. 

```javascript
y_prediksi = model.predict(X_test)
print(y_prediksi)
score = met.accuracy_score(y_test, y_prediksi)
print("accuracy=", score)
precision = met.precision_score(y_test, y_prediksi)
print("precision=", precision)
recall = met.recall_score(y_test, y_prediksi)
print("recall=", recall)
auc = met.roc_auc_score(y_test, y_prediksi)
print("AUC=", auc)
```

Dari hasil diatas, nilai akurasi model adalah 85%. Sebagai upaya membuat model yang kinerjanya lebih baik, kita akan mencoba algoritma lain yaitu Random Forest. Secara umum, algoritma ini lebih cocok untuk digunakan di dataset yang kelasnya tidak seimbang seperti kasus ini. 

### 6. Re-training

```javascript
import sklearn.ensemble as ens
import sklearn.metrics as met
model = ens.RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train, y_train)
y_prediksi = model.predict(X_test)
print(y_prediksi)
score = met.accuracy_score(y_test, y_prediksi)
print("accuracy=", score)
precision = met.precision_score(y_test, y_prediksi)
print("precision=", precision)
recall = met.recall_score(y_test, y_prediksi)
print("recall=", recall)
auc = met.roc_auc_score(y_test, y_prediksi)
print("AUC=", auc)
```
Angka akurasi meningkat menjadi 93,8% . Nilai recall juga meningkat cukup signifikan dibandingkan dengan algoritma Logistic Regression. 

### 7. Melihat Fitur yang Paling Mempengaruhi Churn
Selanjutnya akan dilihat 10 feature yang paling mempengaruhi churn dengan menggunakan atribut feature_importances. Hasilnya akan ditampilkan dalam sebuah bar chart. 

```javascript
important_feature = pd.Series(model.feature_importances_, index=X.columns)
important_feature.nlargest(10).plot(kind='barh')
```
<img src="images/download (2).png?raw=true"/>

### 8. Kesimpulan 

Tiga faktor penentu terpenting terhadap kemungkinan apakah seorang pelanggan akan berhenti berlangganan adalah tenure (lamanya pelanggan menjadi pelanggan), reload (jumlah isi ulang pulsa), dan days active (jumlah hari aktif menggunakan layanan).

Kartu B dan kartu C adalah dua produk yang berpengaruh terhadap churn score. Artinya pelanggan produk kartu A lebih setia dan tidak perlu terlalu dikhawatirkan akan berhenti berlangganan dalam waktu dekat.

Berdasarkan model prediksi yang sudah dibuat, hendaknya pihak departemen marketing perusahaan ini dapat membidik pelanggan yang masuk dalam kriteria di atas agar mereka dapat menjadi target program promosi, untuk mencegah mereka berhenti berlangganan.

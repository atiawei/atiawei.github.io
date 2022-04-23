## Churn Prediction

**Project description:** Dalam contoh kasus ini, akan dibuat model churn prediction untuk sebuah perusahaan telekomunikasi yang menjual layanan internet nirkabel dengan kartu prabayar. Tidak sedikit pelanggan perusahaan ini yang telah berpindah langganan ke perusahaan pesaing akibat tawaran harga dan layanan yang lebih menarik. Pihak manajemen perusahaan ini menyadari terkait masalah tersebut dan berencana meluncurkan program-program promosi untuk menahan churn rate. Program promosi ini hanya akan ditawarkan melalui SMS kepada kelompok pelanggan yang dianggap rawan churn. Agar lebih efektif, machine learning diperlukan untuk menentukan kelompok pelanggan tsb.

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

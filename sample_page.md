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

### Melihat Isi Kolom
```javascript
df1['product'].value_counts()
```
Karena masih berupa teks berisi nama-nama produk ("Kartu A", "Kartu B", "Kartu C"), maka feature ini harus diubah menjadi numerik. Cara yang paling umum adalah dengan metode one-hot encoding, yaitu menuliskan semua nulai yang mungkin muncul menjadi kolom kemudian memberikan nilai 0 dan 1 di kolom tersebut, tergantung pada pelanggan yang diwakili di setiap baris apakah menggunakan produk yang bersangkutan. Dalam kasus kita, karena ada tiga produk maka akan ada tiga kolom baru.

```javascript
pd.get_dummies(df1['product'])
```

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 


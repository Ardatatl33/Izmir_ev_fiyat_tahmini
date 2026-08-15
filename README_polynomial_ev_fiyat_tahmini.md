# Polinomal Regresyon ile Ev Fiyat Tahmini

## Proje Adı

**Polinomal Regresyon ile Ev Fiyat Tahmini**

## Projenin Amacı

Bu projenin amacı, İzmir'deki ev ilanlarından elde edilen özellikleri
kullanarak ev fiyatlarını tahmin etmektir.

Projede ana makine öğrenmesi yöntemi olarak **Polynomial Regression**
kullanılmıştır. Polynomial Regression modelinin sonuçlarını karşılaştırabilmek
için ayrıca bir **Linear Regression** baseline modeli oluşturulmuştur.

Bu çalışma production seviyesinde bir tahmin sistemi kurmak yerine veri
hazırlama, polynomial özellik oluşturma, model eğitme ve model değerlendirme
adımlarını anlaşılır şekilde uygulamayı amaçlamaktadır.

## Veri Seti

Projede `Izmir-House-Prices.csv` veri seti kullanılmıştır. Veri setinde 5.841 kayıt
ve aşağıdaki sütunlar bulunmaktadır:

- `left`: Konut tipi
- `price`: Ev fiyatı
- `room`: Oda sayısı
- `salon`: Salon sayısı
- `area`: Evin metrekaresi
- `age`: Bina yaşı
- `province`: İl
- `district`: İlçe

Modelin tahmin etmeye çalıştığı hedef değişken `price` sütunudur.

`province` sütunundaki bütün kayıtlar İzmir olduğu için bu sütun modele yeni
bir bilgi sağlamamaktadır. Bu nedenle `province` model özelliklerine dahil
edilmemiştir.

## Projede Yapılan İşlemler

Projede aşağıdaki adımlar sırasıyla uygulanmıştır:

1. Veri setinin yüklenmesi ve incelenmesi
2. Eksik veri kontrolü
3. IQR yöntemiyle aykırı değer analizi
4. Sayısal sütunlar için korelasyon analizi
5. Basit öznitelik mühendisliği
6. Kategorik değişkenlerin One-Hot Encoding ile dönüştürülmesi
7. Verinin train, validation ve test kümelerine ayrılması
8. Sayısal özelliklerin StandardScaler ile ölçeklenmesi
9. Linear Regression baseline modelinin eğitilmesi
10. Polynomial Regression modelinin eğitilmesi
11. Polynomial degree 1, 2 ve 3 değerlerinin karşılaştırılması
12. Validation sonuçlarına göre en iyi degree değerinin seçilmesi
13. Seçilen modelin test setinde final değerlendirmesi
14. Gerçek ve tahmin edilen fiyatların karşılaştırılması

Aykırı fiyat analizinde Villa, Yalı, Köşk ve Müstakil gibi konutların doğal
olarak daha pahalı olabileceği dikkate alınmıştır. Bu nedenle son fiyat
temizliği, yeterli sayıda kaydı olan konut türlerinin kendi içindeki IQR
sınırlarına göre yapılmıştır.

## Feature Engineering

Projede oda ve salon sayılarını birlikte değerlendirebilmek için aşağıdaki
yeni özellik oluşturulmuştur:

```python
toplam_oda = room + salon
```

`toplam_oda` özelliği, bir evdeki toplam oda ve salon sayısını temsil eder.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Kullanılan Makine Öğrenmesi Modelleri

- Linear Regression
- Polynomial Regression

Linear Regression modeli karşılaştırma yapabilmek için baseline olarak
kullanılmıştır. Projenin ana modeli Polynomial Regression'dır. Polynomial
özellikler yalnızca `room`, `salon`, `area`, `age` ve `toplam_oda` sayısal
özelliklerine uygulanmıştır. One-Hot Encoding ile oluşan kategorik sütunlar
gereksiz özellik artışını önlemek için polynomial dönüşüme dahil edilmemiştir.

## Model Değerlendirme Metrikleri

Modellerin performansını değerlendirmek için aşağıdaki metrikler
kullanılmıştır:

- **MAE:** Tahminlerin gerçek değerlerden ortalama mutlak sapmasını gösterir.
- **MSE:** Hataların karelerini aldığı için büyük tahmin hatalarını daha fazla cezalandırır.
- **RMSE:** Tahmin hatasının hedef değişkenle aynı birimde yorumlanmasını sağlar.
- **R² Score:** Modelin hedef değişkendeki değişimin ne kadarını açıklayabildiğini gösterir.

MAE, MSE ve RMSE değerlerinin düşük olması; R² değerinin ise yüksek olması
beklenir.

## Polynomial Degree Karşılaştırması

Degree 1, degree 2 ve degree 3 modelleri yalnızca validation seti üzerinde
karşılaştırılmıştır. Test seti model veya degree seçimi için kullanılmamış,
yalnızca final değerlendirme aşamasında açılmıştır.

Kodun gerçek veri setiyle çalıştırılması sonucunda elde edilen validation
sonuçları aşağıdadır:

| Degree | MAE | RMSE | R² | Özellik Sayısı |
|---:|---:|---:|---:|---:|
| 1 | 1.862.067,87 | 3.422.646,78 | 0,6614 | 47 |
| 2 | 1.823.508,61 | 3.352.682,76 | 0,6751 | 62 |
| 3 | 1.804.497,70 | 3.300.357,70 | 0,6852 | 97 |

Validation setinde en yüksek R² ve en düşük RMSE değerini degree 3 verdiği
için final Polynomial Regression modeli için **degree 3** seçilmiştir.

## Final Test Sonuçları

Degree seçimi tamamlandıktan sonra modeller test setinde değerlendirilmiştir.
Gerçek test sonuçları aşağıdadır:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.850.256,76 | 11.078.918.013.962,33 | 3.328.500,87 | 0,6465 |
| Polynomial Regression | 1.946.315,57 | 32.793.972.089.201,26 | 5.726.602,14 | -0,0464 |

Validation setinde degree 3 daha başarılı görünmesine rağmen test setinde
Polynomial Regression modeli aynı başarıyı sürdürememiştir. Linear Regression
modeli daha düşük RMSE ve daha yüksek R² değerine ulaştığı için final test
sonuçlarında daha başarılı olmuştur.

Polynomial Regression modelindeki validation ve test farkı, modelin eğitim
verisine fazla uyum sağlaması veya veri setindeki uç özellik değerlerinden daha
fazla etkilenmesiyle ilişkili olabilir. Bu sonuç, polynomial degree değerini
artırmanın modeli her zaman daha başarılı yapmadığını göstermektedir.

## Proje Yapısı

```text
turk-yeyapayzekaakademisi_bootcamp/
│
├── machine_learning/
│   ├── polynomial_ev_fiyat_tahmini.py
│   ├── Izmir-House-Prices.csv
│   ├── README_polynomial_ev_fiyat_tahmini.md
│   ├── requirements.txt
│   └── .gitignore
```

Sanal ortam klasörü olan `venv/` veya `.venv/` bilgisayarda oluşturulur ve
GitHub'a yüklenmez.

## Projeyi Çalıştırma

İlk olarak repository klasöründe bir sanal ortam oluşturulabilir:

```bash
python -m venv venv
```

Windows üzerinde sanal ortamı etkinleştirmek için:

```bash
venv\Scripts\activate
```

Gerekli kütüphaneleri tek komutla kurmak için:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatif olarak `requirements.txt` dosyası kullanılabilir:

```bash
pip install -r machine_learning/requirements.txt
```

Proje, repository ana klasöründen aşağıdaki komutla çalıştırılabilir:

```bash
python machine_learning/polynomial_ev_fiyat_tahmini.py
```

## Sonuç

Bu projede İzmir'deki ev ilanları kullanılarak Linear Regression ve Polynomial
Regression modelleri karşılaştırılmıştır. Polynomial Regression, sayısal
özellikler arasındaki doğrusal olmayan ilişkileri modelleyebilmek amacıyla
kullanılmış ve degree 1, 2 ve 3 değerleri validation setinde karşılaştırılmıştır.

Validation sonuçlarına göre degree 3 seçilmiş olsa da final testte Linear
Regression daha başarılı olmuştur. Bu durum, daha karmaşık bir modelin her
zaman daha iyi genelleme yapmadığını ve test setinin yalnızca final değerlendirme
için kullanılmasının neden önemli olduğunu göstermektedir.

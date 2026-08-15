# Makine Öğrenmesi ile İzmir Ev Fiyat Tahmini: Linear ve Polynomial Regression

**Python ve Scikit-learn kullanarak İzmir'deki evlerin fiyatlarını Linear Regression ve Polynomial Regression modelleriyle tahmin ettiğim bootcamp final projesi.**

Bir evin fiyatını belirleyen tek bir özellik yok. Metrekare, oda sayısı, bina yaşı, bulunduğu ilçe ve konut tipi gibi birçok değişken fiyat üzerinde birlikte etkili olabiliyor. Üstelik bu değişkenlerin fiyatla ilişkisi her zaman düz bir çizgi gibi ilerlemiyor.

Buradan hareketle şu soruya cevap aradım:

**Bu özelliklerden yararlanarak bir evin fiyatını makine öğrenmesi ile tahmin edebilir miyiz?**

Bu proje, bootcamp boyunca öğrendiğim veri analizi, veri ön işleme, feature engineering, Linear Regression, Polynomial Regression ve model değerlendirme konularını tek bir çalışma içinde uygulama fırsatı verdi. Bu yazıda projenin yalnızca sonucunu değil, veriyi ilk açtığım andan final test sonuçlarına kadar izlediğim yolu paylaşacağım.

## Projenin Amacı

Projenin temel amacı, İzmir'deki evlere ait özellikleri kullanarak ev fiyatını tahmin eden bir regresyon modeli geliştirmekti.

Çalışmada iki temel yaklaşım kullandım:

1. **Linear Regression**
2. **Polynomial Regression**

Linear Regression, karşılaştırma yapabilmek için başlangıç yani baseline modelim oldu. Projenin ana yöntemi ise Polynomial Regression'dı. Ayrıca tek bir polynomial degree değerine bağlı kalmak yerine degree 1, 2 ve 3 modellerini validation setinde karşılaştırdım.

Buradaki önemli nokta, daha karmaşık modeli doğrudan daha iyi kabul etmemekti. Modellerin gerçekten nasıl performans gösterdiğine MAE, MSE, RMSE ve R² metrikleri üzerinden karar verdim.

## Veri Setini Tanıyalım

Projede `Izmir-House-Prices.csv` adlı veri setini kullandım. Veri seti **5.841 satır ve 8 sütundan** oluşuyor.

| Sütun | Açıklama |
|---|---|
| `left` | Konut tipi |
| `price` | Ev fiyatı |
| `room` | Oda sayısı |
| `salon` | Salon sayısı |
| `area` | Evin metrekaresi |
| `age` | Bina yaşı |
| `province` | İl |
| `district` | İlçe |

Modelin tahmin etmeye çalıştığı hedef değişken `price` sütunuydu.

`province` sütununu incelediğimde bütün kayıtların İzmir'e ait olduğunu gördüm. Bir özellik bütün satırlarda aynı değeri taşıyorsa model için ayırt edici bilgi sağlamaz. Bu nedenle `province` sütununu model özelliklerine dahil etmedim.

Eksik değer kontrolü sonucunda sekiz sütunun hiçbirinde eksik değer bulunmadı. Yine de gerçek bir projede veri her zaman eksiksiz gelmeyebileceği için kontrol adımını kodda bıraktım. Eksik değer olsaydı sayısal sütunlarda medyan, kategorik sütunlarda mod kullanılarak basit bir doldurma yapılacaktı.

[BURAYA: Veri setinin ilk satırlarının ekran görüntüsü]

*Şekil 1: Veri setindeki sütunlar ve ilk ev ilanı kayıtları.*

## Veriyi Keşfetmek

Veriyi modele vermeden önce neyle çalıştığımı anlamam gerekiyordu. Bunun için Pandas'ın temel inceleme araçlarını kullandım:

```python
df = pd.read_csv("Izmir-House-Prices.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

`head()` ilk kayıtları, `shape` satır ve sütun sayısını, `info()` veri tiplerini, `describe()` ise sayısal sütunların temel istatistiklerini gösterdi. `isnull().sum()` ile de eksik değerleri sütun bazında kontrol ettim.

İlk istatistiklerde fiyat sütununda dikkat çekici bir durum vardı. Ortalama fiyat yaklaşık 8,7 milyon görünürken maksimum değer 10 milyara ulaşıyordu. En düşük fiyat ise 650'ydi. Bu kadar geniş aralık, aykırı değerleri ayrıca incelemem gerektiğini gösterdi.

## Aykırı Değerlerle Karşılaşmak

Aykırı değer, veri setinin genel yapısından belirgin biçimde uzaklaşan gözlemdir. Ancak ev verilerinde pahalı bir villa veya yalı yalnızca yüksek fiyatlı olduğu için otomatik olarak hatalı kabul edilemez. Bu nedenle temizlik yaparken hem istatistiksel sınırları hem de konut tiplerini dikkate aldım.

İlk analizde IQR yöntemini kullandım. Buradaki temel kavramlar şöyle:

- **Q1:** Verinin yüzde 25'lik noktası
- **Q3:** Verinin yüzde 75'lik noktası
- **IQR:** Q3 ile Q1 arasındaki fark
- **Alt sınır:** Q1'in 1,5 IQR altı
- **Üst sınır:** Q3'ün 1,5 IQR üstü

Kodun temel kısmı oldukça sade:

```python
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR
```

Bütün konutları tek sınırla değerlendiren global IQR kontrolünde **622 aykırı değer adayı** bulundu. Fakat bu yaklaşım pahalı konut türlerini dairelerle aynı dağılım içinde değerlendiriyordu.

Bu nedenle final temizliği, yeterli sayıda kaydı olan her konut tipinin kendi fiyat dağılımı içinde yaptım. Çok az örneği olan konut türlerinde çeyrek değerler güvenilir olmayacağı için otomatik silme uygulamadım.

Gerçek temizlik sonucu:

- Temizleme öncesi satır sayısı: **5.841**
- Global IQR ile bulunan aday sayısı: **622**
- Konut türü bazlı final aykırı değer sayısı: **401**
- Temizleme sonrası satır sayısı: **5.440**

Bu adım özellikle 10 milyar ve 1,785 milyar gibi veri setinin genel yapısını çok fazla etkileyen fiyatların model eğitimine girmesini engelledi. Bununla birlikte, yüksek fiyatlı her kaydı kör şekilde silmedim.

`area` sütununda 20 metrekarenin altında görünen 26 kayıt da vardı. Bazılarının yazım biçiminden kaynaklanan bir sorun olma ihtimali bulunduğu için bu kayıtları ekrana yazdırarak inceledim; anlamları kesin olmadığı için otomatik olarak değiştirmedim veya silmedim.

## Veriyi Grafiklerle İncelemek

Sayısal özetler önemli olsa da dağılımı ve değişkenler arasındaki ilişkiyi grafiklerde görmek veriyi anlamayı kolaylaştırdı.

### Ev Fiyatlarının Dağılımı

Aykırı fiyat temizliğinden sonra medyan fiyat **3.750.000**, ortalama fiyat ise yaklaşık **5.521.766** oldu. Ortalama değerin medyandan yüksek olması ve dağılımın çarpıklık değerinin yaklaşık 3,41 çıkması, fiyatların sağa çarpık bir dağılıma sahip olduğunu gösteriyor. Yani ilanların büyük bölümü daha düşük ve orta fiyat aralıklarında toplanırken az sayıdaki pahalı konut sağ tarafta uzun bir kuyruk oluşturuyor.

[BURAYA: Ev fiyatlarının dağılım grafiği]

*Şekil 2: Aykırı değer temizliğinden sonra ev fiyatlarının sağa çarpık dağılımı.*

### Metrekare ve Ev Fiyatı

`area` ile `price` arasındaki korelasyon yaklaşık **0,4524** olarak hesaplandı. Scatter grafikte metrekare arttıkça fiyatın yükselme eğilimi görülebiliyor; ancak noktalar tek bir çizgi üzerinde toplanmıyor. Aynı metrekareye sahip evlerin ilçe, bina yaşı ve konut tipi gibi nedenlerle farklı fiyatlara sahip olması bu yayılımı açıklıyor.

Bu nedenle grafiği “metrekare fiyatı kesin belirler” şeklinde değil, “metrekare fiyatla orta düzeyde pozitif ilişki gösteren önemli özelliklerden biridir” şeklinde yorumlamak daha doğru.

[BURAYA: Metrekare - fiyat ilişkisi grafiği]

*Şekil 3: Metrekare ile fiyat arasındaki pozitif eğilim ve ilanlar arasındaki yayılım.*

### Korelasyon Matrisi

Korelasyon, iki sayısal değişkenin birlikte nasıl hareket ettiğini özetleyen bir ölçüdür. Temizlenmiş veride `price` ile korelasyonlar şöyleydi:

- `area`: **0,4524**
- `room`: **0,4199**
- `salon`: **0,1900**
- `age`: **-0,1060**

Metrekare ve oda sayısı fiyatla pozitif ilişki gösterirken bina yaşıyla zayıf negatif bir ilişki görüldü. Bu değerler tek başına neden-sonuç ilişkisi kurmaz; yalnızca değişkenlerin birlikte hareket etme yönü ve gücü hakkında fikir verir.

[BURAYA: Korelasyon heatmap grafiği]

*Şekil 4: Sayısal özelliklerin birbirleri ve ev fiyatıyla korelasyonları.*

## Yeni Bir Özellik Üretmek: Toplam Oda

Feature engineering, mevcut verilerden model için anlamlı olabilecek yeni özellikler üretme işlemidir. Bu projede karmaşık dönüşümler yerine basit ve anlaşılır bir özellik oluşturdum:

```python
df_clean["toplam_oda"] = (
    df_clean["room"] + df_clean["salon"]
)
```

`room` ve `salon` ayrı bilgiler olsa da ikisini birlikte ifade eden `toplam_oda` özelliği evin toplam oda yapısını tek sütunda gösteriyor. Bu özellik, özellikle aynı metrekarede farklı oda düzenine sahip evleri ayırmada modele ek bilgi sağlayabilir.

## Makine Öğrenmesi Metinleri Nasıl Anlayacak?

Veri setindeki `left` ve `district` sütunları metinsel değerlerden oluşuyor. Örneğin modelin karşısına Karşıyaka, Bornova, Villa veya Daire gibi ifadeler geliyor.

Linear Regression bu metinleri doğrudan işleyemez. Bu nedenle kategorik değerleri One-Hot Encoding ile sayısal sütunlara dönüştürdüm:

```python
X = pd.get_dummies(
    X,
    columns=["left", "district"],
    drop_first=True,
    dtype=int
)
```

One-Hot Encoding, her kategori için 0 ve 1 değerlerinden oluşan ayrı sütunlar üretir. Örneğin bir ilan Villa ise ilgili Villa sütunu 1, diğer konut tipi sütunları 0 olur. `drop_first=True` kullanarak her kategorik değişkenin ilk sınıfını düşürdüm ve gereksiz tekrarın önüne geçtim.

Bu işlemden sonra model girdisi **47 özelliğe** ulaştı.

## Veriyi Eğitim, Doğrulama ve Test Olarak Ayırmak

Modelin yalnızca gördüğü veriyi ezberleyip ezberlemediğini anlayabilmek için veri setini üç parçaya ayırdım:

- **Train (%60):** Modelin öğrendiği veri
- **Validation (%20):** Modelleri ve polynomial degree değerlerini karşılaştırdığım veri
- **Test (%20):** Seçilen yaklaşımın daha önce görmediği verideki final performansını ölçtüğüm veri

```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42
)
```

Temizlenmiş 5.440 kayıt şu şekilde bölündü:

- Train: **3.264 satır**
- Validation: **1.088 satır**
- Test: **1.088 satır**

Test setini degree veya model seçimi için kullanmadım. Böylece final değerlendirmede modelin gerçekten görmediği bir veri üzerinde nasıl davrandığını ölçebildim.

## Özellikleri Aynı Ölçeğe Getirmek

`room`, `salon`, `area`, `age` ve `toplam_oda` aynı büyüklükte değerler taşımıyor. Örneğin oda sayısı birkaç birimle ifade edilirken metrekare yüzlerce olabilir. Bu sayısal özellikleri ortak bir ölçeğe getirmek için StandardScaler kullandım.

```python
scaler = StandardScaler()

X_train_scaled[sayisal_sutunlar] = scaler.fit_transform(
    X_train[sayisal_sutunlar]
)

X_val_scaled[sayisal_sutunlar] = scaler.transform(
    X_val[sayisal_sutunlar]
)
```

Scaler yalnızca train verisinde `fit_transform()` ile öğrendi. Validation ve test verilerine yalnızca `transform()` uygulandı.

Eğer scaler bütün veriden bilgi öğrenseydi validation ve test setlerinin dağılımı eğitim sürecine sızmış olurdu. Buna **data leakage** denir. Test verisinin eğitim aşamasına bilgi vermemesi, elde edilen sonucun güvenilir olması açısından önemlidir.

## İlk Model: Linear Regression

Linear Regression, özelliklerle hedef değişken arasında doğrusal bir ilişki kurmaya çalışır. Çok matematiksel düşünmeden anlatmak gerekirse model; metrekare, oda sayısı veya bina yaşı değiştiğinde fiyatın nasıl değiştiğini düz bir ilişki üzerinden öğrenmeye çalışır.

Bu projede Linear Regression başlangıç modeliydi:

```python
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

linear_val_pred = linear_model.predict(X_val_scaled)
```

Validation sonuçları:

| Metrik | Sonuç |
|---|---:|
| MAE | 1.862.067,87 |
| MSE | 11.714.511.002.287,84 |
| RMSE | 3.422.646,78 |
| R² | 0,6614 |

Bu model daha sonra Polynomial Regression sonuçlarını karşılaştırabileceğim baseline değerlerini sağladı.

## Doğrusal İlişki Yetmezse: Polynomial Regression

Ev fiyatı ile metrekare arasındaki ilişkinin her zaman tamamen doğrusal olması gerekmez. Örneğin metrekare artışının fiyat üzerindeki etkisi küçük ve büyük evlerde aynı olmayabilir. Oda sayısı ile metrekarenin birlikte etkisi de tek tek etkilerinden farklı davranabilir.

Polynomial Regression, mevcut sayısal özelliklerden yeni terimler üreterek bu tür doğrusal olmayan ilişkileri yakalamaya çalışır. Degree 2 için x ve x² terimleri; degree 3 için bunlara x³ gibi daha karmaşık terimler eklenir. Birden fazla özellik olduğunda özelliklerin birbiriyle çarpımları da oluşabilir.

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_scaled)
```

Bu projede önemli bir teknik karar verdim: PolynomialFeatures dönüşümünü yalnızca `room`, `salon`, `area`, `age` ve `toplam_oda` sayısal özelliklerine uyguladım. One-Hot Encoding ile oluşan çok sayıdaki kategori sütununu polynomial dönüşüme dahil etmedim.

Böylece degree 3 seviyesinde gereksiz özellik patlamasını önledim. Kategorik dummy sütunları polynomial sayısal özelliklere normal halleriyle eklendi.

## Hangi Polynomial Degree Daha Başarılı?

Polynomial model için degree 1, 2 ve 3 değerlerini ayrı ayrı denedim. Her model train setinde eğitildi ve validation setinde karşılaştırıldı.

| Polynomial Degree | MAE | RMSE | R² |
|---:|---:|---:|---:|
| 1 | 1.862.067,87 | 3.422.646,78 | 0,6614 |
| 2 | 1.823.508,61 | 3.352.682,76 | 0,6751 |
| 3 | 1.804.497,70 | 3.300.357,70 | 0,6852 |

[BURAYA: Polynomial degree karşılaştırma sonuçları]

*Şekil 5: Degree 1, 2 ve 3 modellerinin validation metrikleri.*

Validation setinde degree yükseldikçe MAE ve RMSE azaldı, R² ise arttı. En yüksek R² ve en düşük RMSE degree 3 modelinde elde edildi. Bu nedenle test setine hiç bakmadan **degree 3** seçildi.

Buraya kadar degree 3 daha iyi görünüyordu. Ancak model seçimi ile modelin gerçek genelleme başarısı aynı şey değil. Bunun neden önemli olduğunu final test sonuçlarında daha net gördüm.

## Modelin Başarısını Nasıl Ölçtüm?

Projede dört regresyon metriği kullandım.

### MAE — Mean Absolute Error

Tahmin ile gerçek fiyat arasındaki mutlak farkların ortalamasıdır. Hatanın genel büyüklüğünü anlaşılır biçimde gösterir.

### MSE — Mean Squared Error

Hataların karesini aldığı için büyük tahmin hatalarını daha fazla cezalandırır. Bu nedenle birkaç büyük hata MSE değerini belirgin biçimde artırabilir.

### RMSE — Root Mean Squared Error

MSE'nin kareköküdür. Ev fiyatıyla aynı birimde yorumlanabildiği için “model tipik olarak kaç fiyat birimi hata yapıyor?” sorusuna daha yakın bir cevap verir.

### R² Score

Modelin fiyatlardaki değişimin ne kadarını açıklayabildiğini gösterir. 1'e yaklaşması daha yüksek açıklama gücü anlamına gelir. Negatif R² ise modelin test verisinde yalnızca ortalama fiyatı tahmin etmekten bile daha zayıf kaldığını gösterebilir.

Metrikleri Scikit-learn fonksiyonlarıyla şu şekilde hesapladım:

```python
mae = mean_absolute_error(y_val, tahminler)
mse = mean_squared_error(y_val, tahminler)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, tahminler)
```

Bu dört değer birlikte değerlendirildiğinde modelin hem ortalama hata büyüklüğü hem de fiyat değişimini açıklama gücü görülebiliyor.

## Modeli Daha Önce Görmediği Evlerle Test Etmek

Validation sonucunda seçilen degree 3 Polynomial Regression modelini test setinde değerlendirdim.

**Seçilen Polynomial Degree: 3**

| Metrik | Sonuç |
|---|---:|
| MAE | 1.946.315,57 |
| MSE | 32.793.972.089.201,26 |
| RMSE | 5.726.602,14 |
| R² | -0,0464 |

RMSE'nin yaklaşık **5,73 milyon** olması, büyük fiyat hatalarının test sonucunu ciddi biçimde etkilediğini gösteriyor. R² değerinin negatif çıkması ise degree 3 Polynomial Regression modelinin daha önce görmediği test verisine iyi genelleme yapamadığını ortaya koydu.

Validation setinde daha iyi görünen modelin testte bu kadar gerilemesi; yüksek dereceli modelin train ve validation verisinin yapısına fazla uyum sağlaması veya testteki uç oda, alan ve fiyat değerlerinden daha fazla etkilenmesiyle ilişkili olabilir.

## Linear Regression mı Polynomial Regression mı?

Model seçimi aşamasında Polynomial Regression degree 3 validation sonuçlarına göre seçilmişti. Final testte iki modeli aynı test verisi üzerinde karşılaştırdığımda sonuç değişti:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1.850.256,76 | 3.328.500,87 | 0,6465 |
| Polynomial Regression | 1.946.315,57 | 5.726.602,14 | -0,0464 |

Final testte **Linear Regression daha başarılı oldu**. Daha düşük MAE ve RMSE üretirken R² değeri de belirgin biçimde daha yüksekti.

Bu sonuç projenin benim için en öğretici taraflarından biri oldu. Polynomial Regression daha karmaşık ilişkiler kurabildiği için teoride daha esnek bir model. Fakat esneklik her zaman daha iyi genelleme anlamına gelmiyor. Model karmaşıklığı arttıkça verideki gürültüyü veya uç değerleri öğrenme ihtimali de artıyor.

Kısacası, bu veri ve bu ayrım için daha sade olan Linear Regression test verisinde daha güvenilir sonuç verdi.

## Modelin Tahminlerine Yakından Bakalım

Gerçek fiyatları x eksenine, tahmin edilen fiyatları y eksenine yerleştiren bir scatter grafik oluşturdum. Grafikteki kırmızı kesikli diagonal çizgi ideal tahminleri temsil ediyor. Bir nokta bu çizgiye ne kadar yakınsa tahmin gerçek fiyata o kadar yakın.

Polynomial modelin test grafiğinde bazı noktalar çizgiye yakın olsa da özellikle pahalı ilanlarda belirgin sapmalar görülüyor. Noktaların diagonal çizgi çevresinde sıkı biçimde toplanmaması, negatif R² ve yüksek RMSE sonucuyla tutarlı.

[BURAYA: Gerçek fiyat - tahmin edilen fiyat grafiği]

*Şekil 6: Degree 3 Polynomial Regression modelinin gerçek ve tahmin edilen test fiyatları.*

## Örnek Tahminler

Test setindeki ilk tahminlerden bazıları şöyle:

| Gerçek Fiyat | Tahmin Edilen Fiyat |
|---:|---:|
| 2.650.000,00 | 3.521.076,74 |
| 3.650.000,00 | 4.415.259,51 |
| 2.800.000,00 | 4.221.338,62 |
| 11.000.000,00 | 8.237.502,29 |
| 2.250.000,00 | 2.353.290,85 |
| 3.890.000,00 | 2.902.689,13 |
| 6.000.000,00 | 4.659.490,86 |
| 2.300.000,00 | 4.642.503,53 |
| 3.750.000,00 | 5.131.262,28 |
| 32.000.000,00 | 19.435.503,71 |

Örneğin 2,25 milyonluk ilan için 2,35 milyon civarında oldukça yakın bir tahmin yapılırken 32 milyonluk ilanda model yaklaşık 19,44 milyon tahmin etti. Bu örnekler, ortalama metriklerin arkasındaki farklı hata büyüklüklerini daha somut gösteriyor.

## Kullandığım Teknolojiler

- **Python:** Projenin ana programlama dili
- **Pandas:** CSV okuma, veri temizleme ve tablo işlemleri
- **NumPy:** Sayısal işlemler ve RMSE hesabı
- **Matplotlib:** Dağılım ve tahmin grafiklerinin çizilmesi
- **Seaborn:** Histogram ve korelasyon heatmap görselleri
- **Scikit-learn:** Veri bölme, ölçekleme, model eğitimi ve metrik hesaplama

## Bu Proje Bana Ne Kazandırdı?

Bu çalışma, bootcampte ayrı ayrı öğrendiğim konuları tek bir veri seti üzerinde bir araya getirmemi sağladı.

İlk olarak gerçek bir veri setini yalnızca modele vermenin yeterli olmadığını gördüm. Sütunları, veri tiplerini, eksikleri ve temel istatistikleri incelemek; daha sonra karşılaşılabilecek sorunları erkenden fark etmeyi sağlıyor.

Aykırı değerler konusunda da yalnızca formül uygulamanın yeterli olmadığını öğrendim. 10 milyarlık bir değer açıkça incelenmesi gereken bir uçken pahalı bir villa her zaman hatalı veri değil. Alan bilgisi ile istatistiksel yöntemi birlikte düşünmek gerekiyor.

One-Hot Encoding ile metinsel özellikleri modele uygun hale getirdim, `toplam_oda` özelliğiyle basit bir feature engineering adımı uyguladım ve StandardScaler kullanırken data leakage oluşmamasına dikkat ettim.

Train, validation ve test ayrımının neden üç farklı görev taşıdığını sonuçlarda doğrudan gördüm. Degree 3 validation setinde en iyi modeldi; fakat testte Linear Regression'ın gerisinde kaldı. Bu fark, yalnızca validation sonucuna bakıp modelin her yerde aynı performansı göstereceğini varsaymamam gerektiğini öğretti.

Son olarak MAE, RMSE ve R² değerlerini yalnızca ekrana yazdırılan sayılar olarak değil, modelin gerçek davranışını anlatan ölçüler olarak yorumlama pratiği kazandım.

## Sonuç

Bu projede İzmir'deki ev ilanlarından yararlanarak Linear Regression ve Polynomial Regression modelleriyle fiyat tahmini yaptım. Veriyi inceledim, eksik değer kontrolü gerçekleştirdim, fiyat aykırılarını konut türlerini dikkate alarak temizledim, kategorik sütunları One-Hot Encoding ile dönüştürdüm ve toplam oda özelliğini oluşturdum.

Polynomial Regression için degree 1, 2 ve 3 değerlerini validation setinde karşılaştırdım. En iyi validation sonucu degree 3 modelinde elde edildi: RMSE yaklaşık 3,30 milyon, R² ise 0,6852 oldu.

Final testte ise degree 3 Polynomial Regression modeli RMSE 5,73 milyon ve R² -0,0464 sonucuna geriledi. Linear Regression aynı test setinde RMSE 3,33 milyon ve R² 0,6465 ile daha başarılı çıktı.

Bu nedenle projenin temel çıkarımı, karmaşık bir modelin otomatik olarak daha iyi olmadığıdır. Polynomial özellikler doğrusal olmayan ilişkileri yakalama imkânı sağlasa da degree arttıkça modelin genelleme davranışını validation ve test ayrımına dikkat ederek değerlendirmek gerekir.

Bootcamp final projesi olarak bu çalışma, veri hazırlamadan model karşılaştırmasına kadar bir regresyon problemini baştan sona ele almamı ve sonuçlar beklediğim gibi çıkmadığında bunları gizlemek yerine nedenleriyle yorumlamamı sağladı.

## Proje Kodları

Projenin kaynak kodları aşağıdaki GitHub repository'sinde yer alıyor:

https://github.com/Ardatatl33/turk-yeyapayzekaakademisi_bootcamp

Ana proje dosyası:

`machine_learning/polynomial_ev_fiyat_tahmini.py`

Proje README dosyası:

`machine_learning/README_polynomial_ev_fiyat_tahmini.md`

Kodun tamamı, veri hazırlama adımları ve çalıştırma bilgileri repository içindeki `machine_learning` klasöründe incelenebilir.

"""
Polinomal Regresyon ile Ev Fiyat Tahmini

Amaç:
    Ev özelliklerini kullanarak fiyat tahmini yapmak ve Linear Regression
    ile Polynomial Regression modellerini karşılaştırmak.

Proje adımları:
    1. Veri setini okuma ve inceleme
    2. Eksik ve aykırı değer kontrolü
    3. Korelasyon analizi ve temel görselleştirmeler
    4. Basit öznitelik mühendisliği
    5. Veriyi train, validation ve test kümelerine ayırma
    6. Linear Regression başlangıç modelini eğitme
    7. Polynomial degree 1, 2 ve 3 değerlerini karşılaştırma
    8. En iyi degree ile final modeli eğitme
    9. Test sonuçlarını yorumlama
"""

# Gerekli kütüphaneleri ekleme

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Pandas'ın büyük sayıları 1.850257e+06 şeklinde göstermesini engelliyoruz.
pd.set_option("display.float_format", lambda deger: f"{deger:.2f}")


# Veri setini okuma

# CSV dosyası Python dosyasıyla aynı klasörde olmalıdır.
veri_yolu = Path(__file__).resolve().parent / "Izmir-House-Prices.csv"
df = pd.read_csv(veri_yolu)


# Veri setini inceleme

print("\nVERİ SETİNİN İLK 5 SATIRI")
print(df.head())

print("\nVERİ SETİNİN BOYUTU")
print(df.shape)

print("\nVERİ TİPLERİ")
print(df.info())

print("\nTEMEL İSTATİSTİKLER")
print(df.describe())

print("\nEKSİK DEĞERLER")
print(df.isnull().sum())


# Eksik veri kontrolü

sayisal_sutunlar = ["price", "room", "salon", "area", "age"]
kategorik_sutunlar = ["left", "province", "district"]

# Eksik değer varsa sayısal sütunları medyan, kategorik sütunları mod ile dolduruyoruz.
if df.isnull().sum().sum() > 0:
    for sutun in sayisal_sutunlar:
        if df[sutun].isnull().sum() > 0:
            medyan_deger = df[sutun].median()
            df[sutun] = df[sutun].fillna(medyan_deger)

    for sutun in kategorik_sutunlar:
        if df[sutun].isnull().sum() > 0:
            mod_deger = df[sutun].mode()[0]
            df[sutun] = df[sutun].fillna(mod_deger)

    print("\nEksik değerler medyan ve mod ile dolduruldu.")
    print(df.isnull().sum())
else:
    print("\nVeri setinde eksik değer bulunmadı. Doldurma işlemi yapılmadı.")


# Kategorik değerlerin başındaki ve sonundaki boşlukları temizleme

for sutun in kategorik_sutunlar:
    df[sutun] = df[sutun].str.strip()


# Province sütununu kontrol etme

print("\nPROVINCE SÜTUNUNDAKİ DEĞERLER")
print(df["province"].value_counts())
print("Province tek bir değer içerdiği için modele dahil edilmeyecek.")


# Aykırı değer analizi

print("\nSAYISAL SÜTUNLARIN TEMEL İSTATİSTİKLERİ")
print(df[["price", "room", "salon", "area", "age"]].describe())

temizleme_oncesi_veri_sayisi = len(df)

# Önce bütün veri setinde price sütununun IQR sınırlarını inceliyoruz.
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

global_aykiri_maskesi = (
    (df["price"] < alt_sinir)
    | (df["price"] > ust_sinir)
)

print(f"\nGlobal price alt sınırı: {alt_sinir:.2f}")
print(f"Global price üst sınırı: {ust_sinir:.2f}")
print(f"Global IQR ile bulunan aykırı değer adayı: {global_aykiri_maskesi.sum()}")

# Villa, Yalı ve Müstakil gibi konutlar doğal olarak daha pahalı olabilir.
# Bu nedenle son temizliği yeterli verisi olan her konut türü içinde ayrı yapıyoruz.
# 30'dan az örneği olan türlerde çeyrek değerler güvenilir olmayacağı için silme yapmıyoruz.
price_aykiri_maskesi = pd.Series(False, index=df.index)

for konut_tipi in df["left"].unique():
    tip_maskesi = df["left"] == konut_tipi
    tip_fiyatlari = df.loc[tip_maskesi, "price"]

    if len(tip_fiyatlari) >= 30:
        tip_Q1 = tip_fiyatlari.quantile(0.25)
        tip_Q3 = tip_fiyatlari.quantile(0.75)
        tip_IQR = tip_Q3 - tip_Q1

        tip_alt_sinir = tip_Q1 - 1.5 * tip_IQR
        tip_ust_sinir = tip_Q3 + 1.5 * tip_IQR

        tip_aykiri_maskesi = tip_maskesi & (
            (df["price"] < tip_alt_sinir)
            | (df["price"] > tip_ust_sinir)
        )

        price_aykiri_maskesi = price_aykiri_maskesi | tip_aykiri_maskesi

        print(
            f"{konut_tipi} için bulunan aykırı fiyat sayısı: "
            f"{tip_aykiri_maskesi.sum()}"
        )

bulunan_aykiri_deger_sayisi = price_aykiri_maskesi.sum()

df_clean = df.loc[~price_aykiri_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"\nTemizleme öncesi toplam veri sayısı: {temizleme_oncesi_veri_sayisi}")
print(f"Konut türü bazlı bulunan aykırı değer sayısı: {bulunan_aykiri_deger_sayisi}")
print(f"Temizleme sonrası veri sayısı: {len(df_clean)}")

# Çok küçük area değerleri veri giriş hatası olabilir.
# Anlamı kesin olmadığı için bu satırları otomatik olarak silmiyor, yalnızca inceliyoruz.
supheli_alanlar = df_clean[df_clean["area"] < 20]
print(f"\n20 metrekareden küçük görünen kayıt sayısı: {len(supheli_alanlar)}")
print(supheli_alanlar.head(10))


# Korelasyon analizi

korelasyon = df_clean.corr(numeric_only=True)
print("\nSAYISAL SÜTUNLARIN KORELASYONU")
print(korelasyon)

fiyat_korelasyonlari = (
    korelasyon["price"]
    .sort_values(ascending=False)
)

print("\nPRICE İLE KORELASYONLAR")
print(fiyat_korelasyonlari)


# Temel görselleştirmeler

plt.figure(figsize=(8, 5))
sns.histplot(df_clean["price"], bins=30)
plt.title("Ev Fiyatlarının Dağılımı")
plt.xlabel("Fiyat")
plt.ylabel("Ev Sayısı")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df_clean["area"], df_clean["price"], alpha=0.4)
plt.xlabel("Metrekare")
plt.ylabel("Fiyat")
plt.title("Metrekare ve Ev Fiyatı İlişkisi")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(korelasyon, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sayısal Değişkenlerin Korelasyon Haritası")
plt.tight_layout()
plt.show()


# Öznitelik mühendisliği

# Oda ve salonu toplayarak evin toplam yaşam alanı sayısını gösteren bir özellik oluşturuyoruz.
df_clean["toplam_oda"] = df_clean["room"] + df_clean["salon"]

print("\nTOPLAM ODA ÖZELLİĞİ")
print(df_clean[["room", "salon", "toplam_oda"]].head())


# X ve y değişkenlerini oluşturma

y = df_clean["price"]

X = df_clean[
    [
        "room",
        "salon",
        "area",
        "age",
        "toplam_oda",
        "left",
        "district"
    ]
].copy()


# Kategorik değişkenleri One-Hot Encoding ile dönüştürme

X = pd.get_dummies(
    X,
    columns=["left", "district"],
    drop_first=True,
    dtype=int
)

print("\nONE-HOT ENCODING SONRASI İLK 5 SATIR")
print(X.head())
print(f"One-Hot Encoding sonrası özellik sayısı: {X.shape[1]}")


# Train, validation ve test ayrımı

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Kalan %80 verinin %25'i validation olur: toplamda %60 train, %20 validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42
)

print("\nVERİ KÜMELERİNİN BOYUTLARI")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")


# Sayısal özellikleri ölçekleme

model_sayisal_sutunlar = ["room", "salon", "area", "age", "toplam_oda"]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# Scaler yalnızca eğitim verisi üzerinde öğreniyor.
X_train_scaled[model_sayisal_sutunlar] = scaler.fit_transform(
    X_train[model_sayisal_sutunlar]
)

# Validation ve test verisine yalnızca transform uyguluyoruz.
X_val_scaled[model_sayisal_sutunlar] = scaler.transform(
    X_val[model_sayisal_sutunlar]
)

X_test_scaled[model_sayisal_sutunlar] = scaler.transform(
    X_test[model_sayisal_sutunlar]
)


# Linear Regression başlangıç modeli

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

linear_val_pred = linear_model.predict(X_val_scaled)

linear_val_mae = mean_absolute_error(y_val, linear_val_pred)
linear_val_mse = mean_squared_error(y_val, linear_val_pred)
linear_val_rmse = np.sqrt(linear_val_mse)
linear_val_r2 = r2_score(y_val, linear_val_pred)

print("\nLINEAR REGRESSION VALIDATION SONUÇLARI")
print(f"MAE: {linear_val_mae:.2f}")
print(f"MSE: {linear_val_mse:.2f}")
print(f"RMSE: {linear_val_rmse:.2f}")
print(f"R²: {linear_val_r2:.4f}")


# Polynomial degree karşılaştırması

# PolynomialFeatures yalnızca sayısal özelliklere uygulanıyor.
# One-Hot Encoding sütunlarını polynomial yapmak gereksiz özellik üretebilir.
kategorik_dummy_sutunlar = [
    sutun for sutun in X.columns
    if sutun not in model_sayisal_sutunlar
]

derece_sonuclari = []

for derece in [1, 2, 3]:
    poly = PolynomialFeatures(
        degree=derece,
        include_bias=False
    )

    X_train_poly_sayisal = poly.fit_transform(
        X_train_scaled[model_sayisal_sutunlar]
    )

    X_val_poly_sayisal = poly.transform(
        X_val_scaled[model_sayisal_sutunlar]
    )

    poly_sutun_isimleri = poly.get_feature_names_out(model_sayisal_sutunlar)

    X_train_poly = pd.DataFrame(
        X_train_poly_sayisal,
        columns=poly_sutun_isimleri,
        index=X_train_scaled.index
    )

    X_val_poly = pd.DataFrame(
        X_val_poly_sayisal,
        columns=poly_sutun_isimleri,
        index=X_val_scaled.index
    )

    # Kategorik dummy sütunlarını normal halleriyle polynomial özelliklere ekliyoruz.
    X_train_poly = pd.concat(
        [X_train_poly, X_train_scaled[kategorik_dummy_sutunlar]],
        axis=1
    )

    X_val_poly = pd.concat(
        [X_val_poly, X_val_scaled[kategorik_dummy_sutunlar]],
        axis=1
    )

    polynomial_model = LinearRegression()
    polynomial_model.fit(X_train_poly, y_train)

    polynomial_val_pred = polynomial_model.predict(X_val_poly)

    polynomial_val_mae = mean_absolute_error(y_val, polynomial_val_pred)
    polynomial_val_mse = mean_squared_error(y_val, polynomial_val_pred)
    polynomial_val_rmse = np.sqrt(polynomial_val_mse)
    polynomial_val_r2 = r2_score(y_val, polynomial_val_pred)

    derece_sonuclari.append(
        {
            "Degree": derece,
            "MAE": polynomial_val_mae,
            "MSE": polynomial_val_mse,
            "RMSE": polynomial_val_rmse,
            "R2": polynomial_val_r2,
            "Özellik Sayısı": X_train_poly.shape[1]
        }
    )

    print(f"\nPolynomial Degree: {derece}")
    print(f"MAE: {polynomial_val_mae:.2f}")
    print(f"RMSE: {polynomial_val_rmse:.2f}")
    print(f"R²: {polynomial_val_r2:.4f}")
    print(f"Özellik sayısı: {X_train_poly.shape[1]}")


# En iyi polynomial degree değerini seçme

# Önce yüksek R², eşitlik durumunda düşük RMSE dikkate alınıyor.
en_iyi_sonuc = max(
    derece_sonuclari,
    key=lambda sonuc: (sonuc["R2"], -sonuc["RMSE"])
)

en_iyi_derece = en_iyi_sonuc["Degree"]

print(
    "\nValidation sonuçlarına göre seçilen Polynomial Degree: "
    f"{en_iyi_derece}"
)

validation_sonuclari = pd.DataFrame(derece_sonuclari)
print("\nPOLYNOMIAL VALIDATION KARŞILAŞTIRMASI")
print(validation_sonuclari)


# Final model ve test sonuçları

# Degree seçildikten sonra train ve validation verilerini birleştiriyoruz.
# Test verisi bu aşamaya kadar model seçimi veya eğitim için kullanılmadı.
X_final_train = pd.concat([X_train, X_val], axis=0)
y_final_train = pd.concat([y_train, y_val], axis=0)

final_scaler = StandardScaler()

X_final_train_scaled = X_final_train.copy()
X_final_test_scaled = X_test.copy()

X_final_train_scaled[model_sayisal_sutunlar] = final_scaler.fit_transform(
    X_final_train[model_sayisal_sutunlar]
)

X_final_test_scaled[model_sayisal_sutunlar] = final_scaler.transform(
    X_test[model_sayisal_sutunlar]
)


# Final Linear Regression modeli

final_linear_model = LinearRegression()
final_linear_model.fit(X_final_train_scaled, y_final_train)

linear_test_pred = final_linear_model.predict(X_final_test_scaled)

linear_test_mae = mean_absolute_error(y_test, linear_test_pred)
linear_test_mse = mean_squared_error(y_test, linear_test_pred)
linear_test_rmse = np.sqrt(linear_test_mse)
linear_test_r2 = r2_score(y_test, linear_test_pred)


# Final Polynomial Regression modeli

final_poly = PolynomialFeatures(
    degree=en_iyi_derece,
    include_bias=False
)

X_final_train_poly_sayisal = final_poly.fit_transform(
    X_final_train_scaled[model_sayisal_sutunlar]
)

X_final_test_poly_sayisal = final_poly.transform(
    X_final_test_scaled[model_sayisal_sutunlar]
)

final_poly_sutunlari = final_poly.get_feature_names_out(model_sayisal_sutunlar)

X_final_train_poly = pd.DataFrame(
    X_final_train_poly_sayisal,
    columns=final_poly_sutunlari,
    index=X_final_train_scaled.index
)

X_final_test_poly = pd.DataFrame(
    X_final_test_poly_sayisal,
    columns=final_poly_sutunlari,
    index=X_final_test_scaled.index
)

X_final_train_poly = pd.concat(
    [X_final_train_poly, X_final_train_scaled[kategorik_dummy_sutunlar]],
    axis=1
)

X_final_test_poly = pd.concat(
    [X_final_test_poly, X_final_test_scaled[kategorik_dummy_sutunlar]],
    axis=1
)

final_polynomial_model = LinearRegression()
final_polynomial_model.fit(X_final_train_poly, y_final_train)

y_test_pred = final_polynomial_model.predict(X_final_test_poly)

polynomial_test_mae = mean_absolute_error(y_test, y_test_pred)
polynomial_test_mse = mean_squared_error(y_test, y_test_pred)
polynomial_test_rmse = np.sqrt(polynomial_test_mse)
polynomial_test_r2 = r2_score(y_test, y_test_pred)

print("\nFINAL TEST SONUÇLARI")
print(f"Seçilen Polynomial Degree: {en_iyi_derece}")
print(f"MAE: {polynomial_test_mae:.2f}")
print(f"MSE: {polynomial_test_mse:.2f}")
print(f"RMSE: {polynomial_test_rmse:.2f}")
print(f"R²: {polynomial_test_r2:.4f}")


# Linear Regression ve Polynomial Regression karşılaştırması

model_sonuclari = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Polynomial Regression"],
        "MAE": [linear_test_mae, polynomial_test_mae],
        "MSE": [linear_test_mse, polynomial_test_mse],
        "RMSE": [linear_test_rmse, polynomial_test_rmse],
        "R2": [linear_test_r2, polynomial_test_r2]
    }
)

print("\nMODEL TEST SONUÇLARININ KARŞILAŞTIRILMASI")
print(model_sonuclari)


# Gerçek ve tahmin edilen fiyatları karşılaştırma

sonuclar = pd.DataFrame(
    {
        "Gerçek Fiyat": y_test.values,
        "Tahmin Edilen Fiyat": y_test_pred
    }
)

print("\nGERÇEK VE TAHMİN EDİLEN İLK 20 FİYAT")
print(sonuclar.head(20))


# Gerçek fiyat - tahmin grafiği

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)

en_kucuk_fiyat = min(y_test.min(), y_test_pred.min())
en_buyuk_fiyat = max(y_test.max(), y_test_pred.max())

plt.plot(
    [en_kucuk_fiyat, en_buyuk_fiyat],
    [en_kucuk_fiyat, en_buyuk_fiyat],
    color="red",
    linestyle="--",
    label="İdeal Tahmin Çizgisi"
)

plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek ve Tahmin Edilen Ev Fiyatları")
plt.legend()
plt.tight_layout()
plt.show()


# Sonuçları yorumlama

print("\nSONUÇLARIN YORUMU")
print(f"En başarılı polynomial degree: {en_iyi_derece}")

if (
    polynomial_test_r2 > linear_test_r2
    and polynomial_test_rmse < linear_test_rmse
):
    daha_iyi_model = "Polynomial Regression"
elif (
    linear_test_r2 > polynomial_test_r2
    and linear_test_rmse < polynomial_test_rmse
):
    daha_iyi_model = "Linear Regression"
else:
    daha_iyi_model = "Metriklere göre modeller birbirine yakın veya farklı üstünlüklere sahip"

print(f"Test sonuçlarına göre daha iyi model: {daha_iyi_model}")

if polynomial_test_r2 >= 0:
    print(
        f"Polynomial modelin R² değeri {polynomial_test_r2:.4f}. "
        f"Bu değer fiyat değişiminin yaklaşık %{polynomial_test_r2 * 100:.2f} "
        "kadarının model tarafından açıklandığını gösterir."
    )
else:
    print(
        f"Polynomial modelin R² değeri {polynomial_test_r2:.4f}. "
        "Negatif R², modelin ortalama fiyatı tahmin etmekten daha zayıf "
        "kaldığını gösterir."
    )

print(
    f"RMSE değeri {polynomial_test_rmse:.2f}. Bu değer tahminlerin gerçek "
    "fiyatlardan tipik olarak ne kadar saptığını fiyat birimiyle gösterir."
)

if polynomial_test_r2 >= 0.80:
    basari_yorumu = "Model test verisinde yüksek açıklama gücü göstermiştir."
elif polynomial_test_r2 >= 0.50:
    basari_yorumu = "Model test verisinde orta düzeyde açıklama gücü göstermiştir."
elif polynomial_test_r2 >= 0:
    basari_yorumu = "Modelin test verisindeki açıklama gücü sınırlıdır."
else:
    basari_yorumu = "Model test verisinde başarılı bir tahmin performansı gösterememiştir."

print(basari_yorumu)

degree_2_sonucu = next(
    sonuc for sonuc in derece_sonuclari
    if sonuc["Degree"] == 2
)

degree_3_sonucu = next(
    sonuc for sonuc in derece_sonuclari
    if sonuc["Degree"] == 3
)

if (
    degree_3_sonucu["R2"] < degree_2_sonucu["R2"]
    or degree_3_sonucu["RMSE"] > degree_2_sonucu["RMSE"]
):
    print(
        "Degree 3, degree 2'ye göre validation setinde daha kötü sonuç verdi. "
        "Özellik sayısı arttığı için bu durum overfitting ihtimaliyle ilişkili olabilir."
    )
else:
    print(
        "Degree 3'e geçildiğinde validation performansı kötüleşmedi. "
        "Yine de daha yüksek degree değerleri model karmaşıklığını artırır."
    )

if (
    en_iyi_derece == 3
    and polynomial_test_r2 < linear_test_r2
):
    print(
        "Degree 3 validation setinde seçilmiş olsa da test setinde Linear "
        "Regression modelinden daha kötü sonuç verdi. Bu fark, yüksek dereceli "
        "modelin eğitim verisine fazla uyum sağlaması veya uç özellik "
        "değerlerinden daha fazla etkilenmesiyle ilişkili olabilir."
    )

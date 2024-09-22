import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

### EDA ###

# Veri yükleme
file_path = 'C:/Users/Buse/Desktop/Pusula/side_effect_data 1.xlsx'
data = pd.read_excel(file_path)

##check
print(data.head())

##veri boyutları
print(data.shape)

##veri tipleri
print(data.info()) 

##sayısal değişken istatistikleri
print(data.describe())

## eksik veri analizi
print(data.isnull().sum())

print(data['Cinsiyet'].value_counts())
print(data['Kan Grubu'].value_counts())
print(data['Il'].value_counts())


## tarih sütunlarından yeni özellikler oluşturma
data['Ilac_Kullanim_Suresi'] = (data['Ilac_Bitis_Tarihi'] - data['Ilac_Baslangic_Tarihi']).dt.days
data['Ilac_Baslangic_Yasi'] = (data['Ilac_Baslangic_Tarihi'] - data['Dogum_Tarihi']).dt.days / 365.25
data['Yan_Etki_Baslama_Suresi'] = (data['Yan_Etki_Bildirim_Tarihi'] - data['Ilac_Baslangic_Tarihi']).dt.days

## tarih sütunlarını veri setinden çıkarma
data.drop(['Ilac_Baslangic_Tarihi', 'Ilac_Bitis_Tarihi', 'Dogum_Tarihi', 'Yan_Etki_Bildirim_Tarihi'], axis=1, inplace=True)

## sütunları tanımlama
categorical_columns  = ['Cinsiyet', 'Kan Grubu', 'Il', 'Uyruk', 'Ilac_Adi', 'Yan_Etki', 'Alerjilerim', 'Kronik Hastaliklarim', 'Baba Kronik Hastaliklari', 'Anne Kronik Hastaliklari', 'Kiz Kardes Kronik Hastaliklari', 'Erkek Kardes Kronik Hastaliklari'] 
numerical_columns = ['Kilo', 'Boy', 'Ilac_Kullanim_Suresi', 'Ilac_Baslangic_Yasi', 'Yan_Etki_Baslama_Suresi']  # 'kullanıcı_id' eklenmemiştir.


## aykırı veri analizi
def detect_outliers_iqr(df, column):
    Q1 = np.percentile(df[column], 25)  
    Q3 = np.percentile(df[column], 75) 
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR  
    upper_bound = Q3 + 1.5 * IQR  
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

for col in numerical_columns:
    outliers = detect_outliers_iqr(data, col)
    print(f'{col} Değişkeni için Aykırı Değerler:')
    print(outliers)
 
    
## Sayısal değişkenler arasındaki korelasyon matrisi ve ısı haritası
plt.figure(figsize=(12, 8))
correlation_matrix = data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sayısal Değişkenler Arası Korelasyon Matrisi')
plt.show()


## hasta ve aile fertlerinin kronik hastalıkları ile ilgili değişkenler ve veriler daha kullanışlı hale getirildi.
for col in ['Kronik Hastaliklarim', 'Anne Kronik Hastaliklari', 'Baba Kronik Hastaliklari', 
            'Kiz Kardes Kronik Hastaliklari', 'Erkek Kardes Kronik Hastaliklari']:
    data[col] = data[col].astype(str).str.lower() 
    data[col] = data[col].str.split(',') 
    data[col] = data[col].apply(lambda x: [i.strip().replace("'", '') for i in x if i.strip()] if isinstance(x, list) else x)

## explode
hastaliklar_df = data.explode('Kronik Hastaliklarim')
hastaliklar_df = hastaliklar_df.explode('Anne Kronik Hastaliklari')
hastaliklar_df = hastaliklar_df.explode('Baba Kronik Hastaliklari')
hastaliklar_df = hastaliklar_df.explode('Kiz Kardes Kronik Hastaliklari')
hastaliklar_df = hastaliklar_df.explode('Erkek Kardes Kronik Hastaliklari')

categorical_columns = ['Cinsiyet', 'Kan Grubu', 'Il', 'Uyruk', 'Ilac_Adi', 'Yan_Etki', 'Alerjilerim','Kronik Hastaliklarim', 'Anne Kronik Hastaliklari', 'Baba Kronik Hastaliklari', 
                       'Kiz Kardes Kronik Hastaliklari', 'Erkek Kardes Kronik Hastaliklari']

## kategorik değişkenler için bar grafikleri 
for col in categorical_columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(y=hastaliklar_df[col].dropna(), palette='Set2') 
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()




## artık aile fertlerinin kronik hastalıkları arasında ilişkiler kurulabilir
## örnek modelleme 
hastaliklar_df['Kisi_Diyabet'] = hastaliklar_df['Kronik Hastaliklarim'].apply(lambda x: 1 if isinstance(x, str) and 'diyabet' in x else 0)
hastaliklar_df['Anne_Diyabet'] = hastaliklar_df['Anne Kronik Hastaliklari'].apply(lambda x: 1 if isinstance(x, str) and 'diyabet' in x else 0)
hastaliklar_df['Baba_Diyabet'] = hastaliklar_df['Baba Kronik Hastaliklari'].apply(lambda x: 1 if isinstance(x, str) and 'diyabet' in x else 0)
hastaliklar_df['Anne_ve_Baba_Diyabet'] = ((hastaliklar_df['Anne_Diyabet'] == 1) & (hastaliklar_df['Baba_Diyabet'] == 1)).astype(int)

correlation_matrix = hastaliklar_df[['Kisi_Diyabet', 'Anne_Diyabet', 'Baba_Diyabet', 'Anne_ve_Baba_Diyabet']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Korelasyon'}, linewidths=0.5)
plt.title('Diyabet Durumu Korelasyon Matrisi (Kişi - Anne - Baba - Anne ve Baba Diyabet)')
plt.show()




### DATA PRE-PROCESSING ###


## sayısal değişkenler için eksik veri doldurma
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])


## kategorik değişkenler için eksik veri doldurma
from sklearn.tree import DecisionTreeClassifier

for col in categorical_columns:
    data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [x]) 
    data = data.explode(col)

## label encoding 
encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col].astype(str))

## decisionTree
for col in categorical_columns:
    if data[col].isnull().sum() > 0:

        not_null_data = data[data[col].notnull()]
        null_data = data[data[col].isnull()]

        X_train = not_null_data.drop(columns=[col])
        y_train = not_null_data[col]  # Hedef kategorik sütun

        tree_model = DecisionTreeClassifier()
        tree_model.fit(X_train, y_train)

        X_test = null_data.drop(columns=[col])
        data.loc[data[col].isnull(), col] = tree_model.predict(X_test)

print(data.isnull().sum())





# Sayısal veriler için standardization
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])



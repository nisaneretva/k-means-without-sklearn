import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
from itertools import chain

def Accuracy(y, prediction): # score hesaplama
    prediction = list(prediction)
    score = 0

    for i, j in zip(list(y), prediction):
        if i == j:
            score += 1 #scorun başarı durumunda +1 deger alması ve bunun tum verilere bolunması
    return score / len(y)

df1 = pd.read_csv(r"C:\Users\nisan\Desktop\Nisa_Neretva\BM-4.sınıf\temel_ogrenme\odev\odev-2\penguins_size.csv") #dosyanın okunması
df = df1.copy()
label_encoder = preprocessing.LabelEncoder() 
df1['species']= label_encoder.fit_transform(df1['species'])

# print(df.head())
# print(df.columns)
# print(df.describe())
# print(df.info())

#print(df.info()) # 344 adet penguen verimiz var
#print(df.isnull().sum()) # toplam 18 verimizde null değeri vardır

categorical = [var for var in df.columns if df[var].dtype=='O'] # kategorik verilerin bulunması
#print('The categorical variables are :', categorical)
numerical = [var for var in df.columns if df[var].dtype!='O'] # numerik verilerin bulunması
#print('The numerical variables are:', numerical)

df['culmen_length_mm'].fillna(df['culmen_length_mm'].mean(),inplace = True)
df['culmen_depth_mm'].fillna(df['culmen_depth_mm'].mean(),inplace = True)
df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean(),inplace = True)
df['body_mass_g'].fillna(df['body_mass_g'].mean(),inplace = True)
# sex boş olan verileri bir sonraki gözleme göre dolduruyorum.
df['sex'].fillna(method='bfill', inplace=True)
# print(df["sex"])
# print(df.info())
# print(df.isnull().sum())


# BURADA AYKIRI VERİ TESPİTİ YAPILMIŞ VE HİÇBİR ŞEKİLDE AYKIRI VERİYE RASTLANILMAMIŞTIR.
# print("İzin verilen en yüksek :",df['culmen_length_mm'].mean() + 3*df['culmen_length_mm'].std())
# print("İzin verilen en düşük :",df['culmen_length_mm'].mean() - 3*df['culmen_length_mm'].std())

# print(df[(df['culmen_length_mm'] > 60.25285) | (df['culmen_length_mm'] < 27.5909)])


# print("İzin verilen en yüksek :",df['culmen_depth_mm'].mean() + 3*df['culmen_depth_mm'].std())
# print("İzin verilen en düşük :",df['culmen_depth_mm'].mean() - 3*df['culmen_depth_mm'].std())

# print(df[(df['culmen_depth_mm'] > 23.0582) | (df['culmen_depth_mm'] < 11.2440)])

# print("İzin verilen en yüksek :",df['flipper_length_mm'].mean() + 3*df['flipper_length_mm'].std())
# print("İzin verilen en düşük :",df['flipper_length_mm'].mean() - 3*df['flipper_length_mm'].std())

# print(df[(df['flipper_length_mm'] > 242.9771) | (df['flipper_length_mm'] < 158.8532)])

# print("İzin verilen en yüksek :",df['body_mass_g'].mean() + 3*df['body_mass_g'].std())
# print("İzin verilen en düşük :",df['body_mass_g'].mean() - 3*df['body_mass_g'].std())

# print(df[(df['body_mass_g'] > 6600.5935) | (df['body_mass_g'] < 1802.9152)])

df = pd.get_dummies(df, columns=["sex"])
df = pd.get_dummies(df, columns=["island"])

scaler = MinMaxScaler() #normalizasyon
df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:].to_numpy())

train = df.sample(frac = 0.7, random_state = 42) # %70 train
test = df.drop(train.index) # %30 test

y_train = train["species"]  # tahmin edilcek train classlarının tutulması
x_train = train.drop("species", axis = 1) # tahmin edilecek train classlarının drop edilmesi
# print(x_train.index)

y_test = test["species"] # tahmin edilcek test classlarının tutulması
x_test = test.drop("species", axis = 1) # tahmin edilecek test classlarının drop edilmesi
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#------------------------  SINIFLANDIRMA  -----------------------

means = train.groupby(["species"]).mean() # ortalama hesabı
var = train.groupby(["species"]).var() # varyans hesabı
prior = (train.groupby("species").count() / len(train)).iloc[:,1] # h ın ilk olasılığı
classes = train['species'].unique().tolist() # class isimleri
# print(means)
# print(var)
# print(prior)

def predict(train_x):
    global count
    predics = []
    for i in train_x.index: # Verilerin indexini tutar
        #print(train_x.index)
        sinif_olasilik = [] #sınıflar arası olasılıkların tutulması icin
        instance = train_x.loc[i]
        #print("instance",instance)
        for cls in classes: # classlar arasında dolasma
            ozel_olasilik = [] #ozellikler arası olasilikların tutualması icin
            ozel_olasilik.append((prior[cls])) # class sayısının toplam veriye bölümü (prior) eklenmesi
            for i in train_x.columns: # columnlar arasında dolasma
                #print("col",col)
                x = instance[i]
                mean = means[i].loc[cls] # o sınıfa ait columnların ortalama hesabı
                #print(mean)
                #print("mean",mean)
                #print(var[i])
                sta_sap = var[i].loc[cls]**0.5 #standart sapma
                
                #print("cls",cls)
                #(1/(sta.sap * 2pi**0.5)) * (e**(-0.5 (((x-ort)/(sta.sap))**2))
                if sta_sap == 0:
                    sta_sap = 0.1
                #burada olay patlıyor
                olasilik = (np.e ** (-0.5 * ((x - mean)/sta_sap) ** 2)) / (sta_sap * np.sqrt(2 * np.pi)) #sayısal verilerde naive bayes hesabı
                if olasilik == 0:
                    olasilik = 1/len(train)
                    
                ozel_olasilik.append(olasilik)
                #print("ozellik olasiligi",ozel_olasilik)
            top_olasilik = np.prod(ozel_olasilik) # posterior
        
            sinif_olasilik.append(top_olasilik)
        # En buyuk posterior belirlenmesi ve predictiona eklenmesi

        maxi = sinif_olasilik.index(max(sinif_olasilik))  # olasiligi max olan sınıfın indexinin tutulması
        prediction = classes[maxi]
        predics.append(prediction)
    return predics

predict_train = predict(x_train)
predict_test = predict(x_test)
print("\n")
print("Kendi Kodumun Skor Sonucu (Train) : ",Accuracy(y_train, predict_train))
print("Kendi Kodumun Skor Sonucu (Test)  : ",Accuracy(y_test, predict_test))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
 
print("Hazır Kütüphane Skor Sonucu  : ",nb.score(x_test,y_test))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict_test)

print('Confusion matrix:  \n',cm)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
precision = TP / float(TP + FP)
print("Precision :  ", precision)
recall = TP / float(TP + FN)
print("Recall :  ", recall)
f1score = ((precision * recall) / (precision + recall)) * 2 
print("F1 Score :  ", f1score)


#------------------------  KÜMELEME  -----------------------

dictionary={}
count=0
true_class = 0
false_class = 0

adelie = df[df["species"]=="Adelie"].sample(n=1)
chinstrap = df[df["species"]=="Chinstrap"].sample(n=1)
gentoo = df[df["species"]=="Gentoo"].sample(n=1)

clusters = pd.concat([adelie, chinstrap, gentoo], axis = 0)
#print(clusters)
counter=0
liste_end=[0]
max_iter=100

def knn_hesapla(clusters,df):

# FONKSİYON İLK KISMI
    global counter
    counter+=1
    clusters_without_species = clusters.drop("species", axis = 1)
    #print(clusters_without_species)
    df = df.drop("species", axis = 1)
    min_oklid=[]
    for k in range(len(df)):
        for i in range(len(clusters_without_species)):
            distance = 0.0
            for j in range(len(clusters_without_species.columns)):
            
                distance += (clusters_without_species.iloc[i,j]-df.iloc[k,j])**2

            # i = clusters_without_species.index.values[i]
            # print(t)
            dictionary[i] = distance**0.5
            # indekse oklid uzaklıgının atılması
        import itertools
        sorted_dic = {k: v for k, v in sorted(dictionary.items(), key = lambda v: v[1])}
        min_oklid_dict = dict(itertools.islice(sorted_dic.items(), 1))
    
        min_oklid.append(list(min_oklid_dict.keys()))
        
    
    flatten_list = list(chain.from_iterable(min_oklid))

    se = pd.Series(flatten_list)
    df['species'] = se.values
    #print(df.head(5))
    #print("Sonuccxc",round(Accuracy(df1["species"].values,df["species"].values),5))
    
# FONKSİYON İKİNCİ KISMI
    counter_true = 0
    counter_false = 0
    for i in range(len(df["species"])):
        if df1["species"].values[i] == df["species"].values[i]:
            counter_true+=1
        else:
            counter_false+=1

    kmeans_score = counter_true / (counter_false+counter_true)
   
    uz=len(liste_end)

    if(uz%2==0):
        temp=liste_end[-1]
    else:
        temp=liste_end[(uz-2)]

    if(temp==kmeans_score or counter==max_iter):
        print("Kmeans Clustering Kodumun Skor Sonucu :  ",kmeans_score)
        exit()
       
    liste_end.append(kmeans_score)
    
    new_point(df)

def new_point(df):
    listele = df["species"].unique()
    
    clusters=pd.DataFrame()
    for i in range(0,3):
        new_kume=df[df["species"]==i]
        new_kume_ort = new_kume.mean()
        x=new_kume_ort.to_frame()
        x = x.transpose()
        clusters = pd.concat([x,clusters], axis = 0)

    #print(clusters)
    knn_hesapla(clusters,df)

knn_hesapla(clusters, df)


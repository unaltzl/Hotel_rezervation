################################################
# End-to-End Hotel Machine Learning Pipeline
################################################

import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate, GridSearchCV,
    cross_val_predict, validation_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', None)
pd.set_option("display.expand_frame_repr", False)


df_ = pd.read_excel("PROJE/HotelExcel.xlsx")
df = df_.copy()
df.head()

################################################
# Helper Functions
################################################

# Data Preprocessing & Feature Engineering

def grab_col_names(dataframe, cat_th=7, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def label_encoder(dataframe, binary_cols):
    le = LabelEncoder()
    dataframe[binary_cols] = le.fit_transform(dataframe[binary_cols])
    return dataframe

def hotel_data_prep(dataframe):

    # Feature Engineering
    # Meal Plan Kategorizasyonu:
    ## 0: kahvaltı, 1: akşam yemeği, 2: kupona özel yemek, 3: not selected
    dataframe["type_of_meal_plan"] = dataframe["type_of_meal_plan"].astype(str).replace("Meal Plan 1", 0) \
        .replace("Meal Plan 2", 1).replace("Meal Plan 3", 2).replace("Not Selected", 3)

    dataframe["meal_type_categorized"] = dataframe["type_of_meal_plan"].astype(int)

    dataframe.drop("type_of_meal_plan", inplace=True, axis=1)

    # Room Type Kategorizasyonu:
    # en çok tercih edilen RoomType1=7 en az tercih edilen RoomType3=0
    dataframe["room_type_reserved"] = dataframe["room_type_reserved"].astype(str).replace("Room_Type 3", 0) \
        .replace("Room_Type 7", 1).replace("Room_Type 5", 2).replace("Room_Type 2", 3).replace("Room_Type 6", 4) \
        .replace("Room_Type 4", 5).replace("Room_Type 1", 6)

    dataframe["room_type_categorized"] = dataframe["room_type_reserved"].astype(int)

    dataframe.drop("room_type_reserved", inplace=True, axis=1)

    # Rezervasyon total fiyatı
    dataframe['total_price'] = dataframe['avg_price_per_room'] * (dataframe['no_of_weekend_nights'] + dataframe['no_of_week_nights'])

    # Haftaiçi haftasonu ortalama oda fiyatı
    dataframe["weekend_price"] = dataframe["no_of_weekend_nights"] * dataframe["avg_price_per_room"]
    dataframe["weekday_price"] = dataframe["no_of_week_nights"] * dataframe["avg_price_per_room"]

    # Rezervasyon süresi
    dataframe['reservation_duration'] = dataframe['no_of_weekend_nights'] + dataframe['no_of_week_nights']

    # Total misafir sayısı
    dataframe['total_guests'] = dataframe['no_of_adults'] + dataframe['no_of_children']
    dataframe['avg_price_by_month'] = dataframe.groupby(['arrival_month', "total_guests"])['avg_price_per_room'].transform('mean')
    dataframe["price_different"] = dataframe["avg_price_per_room"] - dataframe["avg_price_by_month"]

    # Rezervasyonun hangi mevsimde yapıldığı bilgisi
    dataframe["arrival_season"] = dataframe["arrival_month"]
    dataframe["arrival_season"].replace([12, 1, 2], "Winter", inplace=True)
    dataframe["arrival_season"].replace([3, 4, 5], "Spring", inplace=True)
    dataframe["arrival_season"].replace([6, 7, 8], "Summer", inplace=True)
    dataframe["arrival_season"].replace([9, 10, 11], "Autumn", inplace=True)

    # Adult ve Child sayısını baskılama
    dataframe.loc[dataframe["no_of_children"] > 2, "no_of_children"] = 3
    dataframe.loc[dataframe["no_of_adults"] > 3, "no_of_adults"] = 3
    # 2017'ye ait olan veriler 2018 olarak değiştirildi
    dataframe["arrival_year"] = dataframe["arrival_year"].replace(2017, 2018)
    # arrival_month, arrival_date, arrival_year kolonları kullanılarak tarih kolonu oluşturuldu.
    dayx = dataframe[["arrival_month", "arrival_date", "arrival_year"]]
    dataframe["Date"] = [str(row[0]) + "-" + str(row[1]) + "-" + str(row[2]) for row in dayx.values]
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors='coerce', format='%m-%d-%Y')

    # müşterilerin kaç gün kaldığı hesaplandı
    dataframe['reservation_duration'] = dataframe['no_of_weekend_nights'] + dataframe['no_of_week_nights']
    # müşterilerin otelden ayrılış tarihleri hesaplandı
    dataframe['arrival_date'] = pd.to_timedelta(dataframe['arrival_date'], unit='D')
    # timedelta kullanıldığı için days çıktısı geliyordu onu alttaki kodlar ile güzelttik.
    dataframe['arrival_date'] = dataframe['arrival_date'].astype(str)
    dataframe['arrival_date'] = dataframe['arrival_date'].str.split().str[0].astype(int)




    #Otele giriş günü
    def create_date_features(dataframe):
        dataframe["Day_of_week_arrival"] = dataframe.Date.dt.dayofweek
        dataframe["is_wknd"] = dataframe.Date.dt.weekday // 5
        return dataframe

    create_date_features(dataframe)
    # The day of the week with Monday=0, Sunday=6.



    # Misafir başına düşen ortalama fiyatı hesapla???
    dataframe['avg_price_per_guest'] = dataframe['avg_price_per_room'] / (dataframe['no_of_adults'] + dataframe['no_of_children'])


    dataframe.columns = [col.upper() for col in dataframe.columns]


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=7, car_th=20)

    cat_cols = [col for col in cat_cols if "BOOKING_STATUS" not in col]


    # LABEL & ONE-HOT ENCODING

    ##### ???? aralıktaki 10 yerine 7 mi olmalı????
    ohe_cols = [col for col in cat_cols if ((7 >= dataframe[col].nunique() > 2) and
                                                     (col not in ['ROOM_TYPE_CATEGORIZED', 'MEAL_TYPE_CATEGORIZED',
                                                                  'NO_OF_ADULTS', 'NO_OF_CHILDREN',
                                                                  'NO_OF_SPECIAL_REQUESTS', 'NO_OF_WEEKEND_NIGHTS']))]

    binary_cols = [col for col in cat_cols if
                   dataframe[col].dtype not in [int, float] and (dataframe[col].nunique() == 2) and (
                               col != "BOOKING_STATUS")]


    for col in binary_cols:
        label_encoder(dataframe, col)


    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtype in ['int64', 'float64'] and col != 'BOOKING_STATUS']

    dataframe = one_hot_encoder(dataframe, ohe_cols)
    return dataframe
df = hotel_data_prep(df)
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove('DATE')

# standartlastirma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

X = df.drop(["BOOKING_STATUS","DATE","BOOKING_ID","ARRIVAL_YEAR",'REPEATED_GUEST'], axis=1)
# Extract target variable
y = df["BOOKING_STATUS"]
# Base Models
def base_models(X, y, scoring="f1"):
    print("Base Models....")
    classifiers = [('LightGBM', LGBMClassifier(random_state= 15)),]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
base_models(X, y)
# f1: 0.819 (LightGBM)

######################
# final_model fonksiyonu
def final_model(X, y):
    print("Final Model")

    LGBMModel = LGBMClassifier(random_state= 15)
    LGBMModel_final = LGBMModel.set_params(colsample_bytree=0.7,
                                           max_depth=10,
                                           n_estimators=500,
                                           num_leaves=50,
                                           random_state=17,
                                           subsample=0.5).fit(X, y)

    scorings = ["f1"]

    for scoring in scorings:
        cv_results = cross_validate(LGBMModel_final, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} (LGBMClassifier)")
    return LGBMModel_final
final_model(X, y)
#f1: 0.8405 (LGBMClassifier)

# plot_importance fonksiyonu
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances1.png")
# final_model fonksiyonunu çağırarak modeli al
final_model1 = final_model(X, y,)
# plot_importance fonksiyonunu çağırarak modeli ve veri kümesini geçir
plot_importance(final_model1, X,save =True)







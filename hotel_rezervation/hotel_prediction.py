################################################
# End-to-End Hotel Machine Learning Pipeline
################################################

import joblib
import pandas as pd

df_ = pd.read_excel("PROJE/HotelExcel.xlsx")
df = df_.copy()

random_user = df.sample(1, random_state=45)

# Modeli kaydetme
joblib.dump(final_model1, 'lgbm_model.joblib')

# Modeli yükleme
loaded_model = joblib.load('lgbm_model.joblib')

# Yüklenen modeli kullanma
random_user = X.sample(1, random_state=50)
prediction = loaded_model.predict(random_user)
prediction
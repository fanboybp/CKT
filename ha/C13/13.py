import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Abalone.data vào một DataFrame
df=pd.read_csv('E:\\Tien_Tien\\DC\\C13\\heart_failure_clinical_records_dataset.csv')
#df.drop(['ID'], axis=1, inplace=True) # bỏ cột ID
# Bước 2: Chuẩn bị dữ liệu cho việc uấn luyện mô hình
X = df[['creatinine_phosphokinase', 'ejection_fraction', 'high_blood_pressure', 'platelets','sex']]  # Đặc trưng
y = df['age']  # tuổi

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Bước 3: Xây dựng và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
# Bênh nhân tuổi trên 60 mà có creatinine_phosphokinase dưới 300
dem = df.loc[(df['age'] > 60) & (df['creatinine_phosphokinase'] < 300)].shape[0]
# Bênh nhân nữ tuổi trên 60
dem2 = df.loc[(df['age'] > 60) & (df['sex'] == 0)].shape[0]
print("Mean Absolute Error:", mae)  #sai số tuyệt đối trung bình
print("Mean Squared Error:", mse)   #Sai số bình phương trung bình
print("Median Absolute Error:", medae) #Sai số tuyệt đối trung vị
# Bênh nhân tuổi trên 60 mà có creatinine_phosphokinase dưới 300
print("Số người dưới 30 tuổi mà có thu nhập trên 50000k $ :"+str(dem))
# Bênh nhân nữ tuổi trên 60
print("Số người dưới 30 tuổi mà có kinh nhiệm trên 5 năm :"+str(dem2))

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': df.min(), 'Max': df.max(), 'Mean': df.mean()}, index=['age', 'creatinine_phosphokinase', 'ejection_fraction','high_blood_pressure','platelets'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('E:\\Tien_Tien\\DC\\C13\\ketqua.txt','w',encoding='utf-8') as file:
    file.write("Bênh nhân tuổi trên 60 mà có creatinine_phosphokinase dưới 300: " + str(dem) + "\n")
    file.write("Bênh nhân nữ tuổi trên 60: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tuổi vs creatinine_phosphokinase
axs[0, 0].scatter(df["age"], df["creatinine_phosphokinase"], c='brown')
axs[0, 0].set_title('Tuổi vs creatinine_phosphokinase')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('creatinine_phosphokinase')

# Biểu đồ 2: Tuổi vs phân suất tống máu
axs[0, 1].scatter(df["age"], df["ejection_fraction"], c='green')
axs[0, 1].set_title('Tuổi vs phân suất tống máu')
axs[0, 1].set_xlabel('Tuổi (Age)')
axs[0, 1].set_ylabel('Phân suất tống máu (ejection_fraction)')

# Biểu đồ 3: Tuổi vs Huyết áp cao
axs[1, 0].scatter(df["age"], df["high_blood_pressure"], c='yellow')
axs[1, 0].set_title('Tuổi vs Huyết áp cao')
axs[1, 0].set_xlabel('Tuổi (Age)')
axs[1, 0].set_ylabel('Huyết áp cao  (high_blood_pressure)')

# Biểu đồ 4: Huyết áp cao với phân suất tống máu
axs[1, 1].scatter(df["high_blood_pressure"], df["ejection_fraction"], c='red')
axs[1, 1].set_title('Huyết áp cao với phân suất tống máu')
axs[1, 1].set_xlabel('Huyết áp cao (high_blood_pressure)')
axs[1, 1].set_ylabel('Phân suất tống máu (ejection_fraction)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
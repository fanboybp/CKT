import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file hondao.csv vào một DataFrame
df = pd.read_csv('E:\\Tien_Tien\\DC\\C12\\Cleveland_hd.csv')

# Xử lý các giá trị thiếu (NaN)
df.fillna(df.mean(), inplace=True)

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']]  # Đặc trưng
y = df['age']  # Nhãn (tuổi)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Bước 3: Xây dựng và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# In kết quả đánh giá
print("Mean Absolute Error:", mae)  # Sai số tuyệt đối trung bình
print("Mean Squared Error:", mse)   # Sai số bình phương trung bình
print("Median Absolute Error:", medae) # Sai số tuyệt đối trung vị

# Bênh nhân tuổi trên 60 mà có huyết áp (trestbps ) trên 160
dem = df.loc[(df['age'] > 60) & (df['trestbps'] > 160)].shape[0]
# Bênh nhân nữ tuổi trên 60
dem2 = df.loc[(df['sex'] == 1) & (df['age'] > 60)].shape[0]

print("Số người trên 60 tuổi có trestbps > 160:", dem)
print("Số người nam trên 60 tuổi:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({
    'Min': df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].min(),
    'Max': df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].max(),
    'Mean': df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mean()
})

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file ketqua.txt
with open('E:\\Tien_Tien\\DC\\C12\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Số người trên 60 tuổi có trestbps > 160: " + str(dem) + "\n")
    file.write("Số người nam trên 60 tuổi: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Biểu đồ 1: age vs trestbps
axs[0, 0].scatter(df["age"], df["trestbps"], c='brown')
axs[0, 0].set_title('Tuổi vs Trestbps')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('Trestbps')

# Biểu đồ 2: age vs chol
axs[0, 1].scatter(df["age"], df["chol"], c='green')
axs[0, 1].set_title('Tuổi vs Cholesterol')
axs[0, 1].set_xlabel('Tuổi (Age)')
axs[0, 1].set_ylabel('Cholesterol')

# Biểu đồ 3: age vs thalach
axs[1, 0].scatter(df["age"], df["thalach"], c='yellow')
axs[1, 0].set_title('Tuổi vs Thalach')
axs[1, 0].set_xlabel('Tuổi (Age)')
axs[1, 0].set_ylabel('Thalach')

# Biểu đồ 4: oldpeak vs chol
axs[1, 1].scatter(df["oldpeak"], df["chol"], c='red')
axs[1, 1].set_title('Oldpeak vs Cholesterol')
axs[1, 1].set_xlabel('Oldpeak')
axs[1, 1].set_ylabel('Cholesterol')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
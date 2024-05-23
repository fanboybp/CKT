import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV và bỏ cột 'GallusID'
df = pd.read_csv('E:\\Tien_Tien\\DC\\C6\\Sample.csv')
df.drop(['MODEL'], axis=1, inplace=True)

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION*']]
y = df['CO2_EMISSIONS']  # khí thải

# Sử dụng One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành các đặc trưng số
#X = pd.get_dummies(X, columns=['GallusBreed', 'GallusEggColor', 'GallusCombType', 'GallusClass', 'GallusLegShanksColor', 'GallusBeakColor', 'GallusEarLobesColor', 'GallusPlumage'])

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

# Lấy giá trị của Mẫu xe có 4 xi lanh và có phát thải CO2 trên 200
dem = df.loc[(df['CYLINDERS'] ==4) & (df['CO2_EMISSIONS'] > 200)].shape[0]
# Lấy giá trị của Mẫu xe có kích cỡ động cơ dưới 2.0 mà tiêu thụ nhiên liệu dưới 9
dem2 = df.loc[(df['ENGINE_SIZE'] < 2.0) & (df['FUEL_CONSUMPTION*'] < 9.0)].shape[0]

print("Mẫu xe có 4 xi lanh và có phát thải CO2 trên 200:", dem)
print("Mẫu xe có kích cỡ động cơ dưới 2.0 mà tiêu thụ nhiên liệu dưới 9:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = df[['ENGINE_SIZE', 'CYLINDERS', 'FUEL_CONSUMPTION*', 'CO2_EMISSIONS']].agg(['min', 'max', 'mean'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file
with open('E:\\Tien_Tien\\DC\\C6\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Mẫu xe có 4 xi lanh và có phát thải CO2 trên 200: " + str(dem) + "\n")
    file.write("Mẫu xe có kích cỡ động cơ dưới 2.0 mà tiêu thụ nhiên liệu dưới 9: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Lượng khí thải CO2 với Kích thước động cơ
axs[0, 0].scatter(df["CO2_EMISSIONS"], df["ENGINE_SIZE"], c='brown')
axs[0, 0].set_title('Lượng khí thải CO2 với Kích thước động cơ')
axs[0, 0].set_xlabel('CO2_EMISSIONS')
axs[0, 0].set_ylabel('ENGINE_SIZE')

# Biểu đồ 2: Lượng khí thải CO2 vs Sự tiêu thụ xăng dầu
axs[0, 1].scatter(df["CO2_EMISSIONS"], df["FUEL_CONSUMPTION*"], c='green')
axs[0, 1].set_title('Lượng khí thải CO2 vs Sự tiêu thụ xăng dầu')
axs[0, 1].set_xlabel('CO2_EMISSIONS')
axs[0, 1].set_ylabel('FUEL_CONSUMPTION*')

# Biểu đồ 3: Lượng khí thải CO2 vs XiLanh
axs[1, 0].scatter(df["CO2_EMISSIONS"], df["CYLINDERS"], c='yellow')
axs[1, 0].set_title('Lượng khí thải CO2 vs XiLanh')
axs[1, 0].set_xlabel('CO2_EMISSIONS')
axs[1, 0].set_ylabel('CYLINDERS')

# Biểu đồ 4: Xilanh với sự tiêu thụ xăng dầu
axs[1, 1].scatter(df["CYLINDERS"], df["FUEL_CONSUMPTION*"], c='red')
axs[1, 1].set_title('Xilanh với sự tiêu thụ xăng dầu')
axs[1, 1].set_xlabel('CYLINDERS')
axs[1, 1].set_ylabel('FUEL_CONSUMPTION*')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
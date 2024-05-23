import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('E:\\Tien_Tien\\DC\\C4\\car_price.csv')
df.drop(['make_model'], axis=1, inplace=True) # Bỏ cột 'make_model' nếu có

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['body_type', 'Body Color', 'km', 'hp', 'Gearing Type', 'Extras']]  # Đặc trưng
y = df['price']  # Mục tiêu

# Sử dụng One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành các đặc trưng số
X = pd.get_dummies(X, columns=['body_type', 'Body Color', 'Gearing Type', 'Extras'])

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

# Đếm số lượng xe ô tô Audi mà có giá trên 20000
dem = df.loc[(df['price'] > 20000)].shape[0]

# Đếm số lượng xe ô tô Audi có màu đỏ loại Compact
dem2 = df.loc[(df['body_type'] == 'Compact') & (df['Body Color'] == 'Red')].shape[0]

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = df[['price', 'km', 'hp']].agg(['min', 'max', 'mean']).transpose()
print("Số xe ô tô Audi mà có giá trên 20000: " + str(dem))
print("Số xe ô tô Audi có màu đỏ loại Compact: " + str(dem2))
print(summary_table)
# Ghi các kết quả vào tệp
with open('E:\\Tien_Tien\\DC\\C4\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Mean Absolute Error: " + str(mae) + "\n")
    file.write("Mean Squared Error: " + str(mse) + "\n")
    file.write("Median Absolute Error: " + str(medae) + "\n\n")
    file.write("Số xe ô tô Audi mà có giá trên 20000: " + str(dem) + "\n")
    file.write("Số xe ô tô Audi có màu đỏ loại Compact: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table.to_string())

# Bước 5: Biểu diễn các quan hệ dữ liệu bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Giá với HP
axs[0, 0].scatter(df['price'], df['hp'], c='brown')
axs[0, 0].set_title('Giá với HP')
axs[0, 0].set_xlabel('Giá (price)')
axs[0, 0].set_ylabel('HP')

# Biểu đồ 2: Giá với KM
axs[0, 1].scatter(df['price'], df['km'], c='green')
axs[0, 1].set_title('Giá với KM')
axs[0, 1].set_xlabel('Giá (Price)')
axs[0, 1].set_ylabel('KM')

# Biểu đồ 3: KM với HP
axs[1, 0].scatter(df['km'], df['hp'], c='yellow')
axs[1, 0].set_title('KM với HP')
axs[1, 0].set_xlabel('KM')
axs[1, 0].set_ylabel('HP')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
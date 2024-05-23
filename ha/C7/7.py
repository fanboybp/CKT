import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV và bỏ cột 'GallusID'
df = pd.read_csv('E:\\Tien_Tien\\DC\\C7\\ENB2012_data.csv')

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['X2', 'X3', 'X4','X5','X6','X7','X8','Y1','Y2']]
y = df['X1']  # khí thải

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

# Mẫu xe có Y1 <15 và có Y2 <26
dem = df.loc[(df['Y1'] <15) & (df['Y2'] <26)].shape[0]
# Mẫu xe có Y1 <15 và X1 <0.7
dem2 = df.loc[(df['Y1'] < 15) & (df['X1'] < 0.7)].shape[0]

print("Mẫu xe có Y1 <15 và có Y2 <26:", dem)
print("Mẫu xe có Y1 <15 và X1 <0.7:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = df[['X1', 'X2', 'X3', 'X4','X5','Y1','Y2']].agg(['min', 'max', 'mean'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file
with open('E:\\Tien_Tien\\DC\\C7\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Mẫu xe có Y1 <15 và có Y2 <26: " + str(dem) + "\n")
    file.write("Mẫu xe có Y1 <15 và X1 <0.7: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tải sửa ấm với độ nén tương đối
axs[0, 0].scatter(df["Y1"], df["X1"], c='brown')
axs[0, 0].set_title('Tải sửa ấm với độ nén tương đối')
axs[0, 0].set_xlabel('Tải sửa ấm (Y1)')
axs[0, 0].set_ylabel('Độ nén tương đối (X1)')

# Biểu đồ 2: Tải sửa ấm với diện tích bề mặt
axs[0, 1].scatter(df["Y1"], df["X2"], c='green')
axs[0, 1].set_title('Tải sửa ấm với diện tích bề mặt')
axs[0, 1].set_xlabel('Tải sửa ấm (Y1)')
axs[0, 1].set_ylabel('Diện tích bề mặt (X2)')

# Biểu đồ 3: Tải làm mát với độ nén tương đối
axs[1, 0].scatter(df["Y2"], df["X1"], c='yellow')
axs[1, 0].set_title('Tải làm mát với độ nén tương đối')
axs[1, 0].set_xlabel('Tải làm mát (Y2)')
axs[1, 0].set_ylabel('Độ nén tương đối (X1)')

# Biểu đồ 4: Tải làm mát với diện tích bề mặt
axs[1, 1].scatter(df["Y2"], df["X2"], c='red')
axs[1, 1].set_title('Tải làm mát với diện tích bề mặt')
axs[1, 1].set_xlabel('Tải làm mát (Y2)')
axs[1, 1].set_ylabel('Diện tích bề mặt (X2)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
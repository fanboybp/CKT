import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('E:\\Tien_Tien\\DC\\C8\\Fish.csv')

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height']]  # Đặc trưng
y = df['Width']  # Nhãn

# Sử dụng One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành các đặc trưng số
X = pd.get_dummies(X, columns=['Species'])

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

# Lấy giá trị của các loại cá 'Bream' có trọng lượng lớn hơn 600
dem = df.loc[(df['Species'] == 'Bream') & (df['Weight'] > 600)].shape[0]
# Lấy giá trị của các loại cá 'Parkki' có trọng lượng lớn hơn 150
dem2 = df.loc[(df['Species'] == 'Parkki') & (df['Weight'] > 150)].shape[0]

print("Số cá 'Bream' có trọng lượng lớn hơn 600:", dem)
print("Số cá 'Parkki' có trọng lượng lớn hơn 150:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({
    'Min': df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].min(),
    'Max': df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].max(),
    'Mean': df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].mean()
})

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file
with open('E:\\Tien_Tien\\DC\\C8\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Số cá 'Bream' có trọng lượng lớn hơn 600: " + str(dem) + "\n")
    file.write("Số cá 'Parkki' có trọng lượng lớn hơn 150: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Trọng lượng vs Chiều rộng
axs[0, 0].scatter(df["Weight"], df["Width"], c='brown')
axs[0, 0].set_title('Trọng lượng vs Chiều rộng')
axs[0, 0].set_xlabel('Trọng lượng (Weight)')
axs[0, 0].set_ylabel('Chiều rộng (Width)')

# Biểu đồ 2: Trọng lượng vs Chiều dài 1
axs[0, 1].scatter(df["Weight"], df["Length1"], c='green')
axs[0, 1].set_title('Trọng lượng vs Chiều dài 1')
axs[0, 1].set_xlabel('Trọng lượng (Weight)')
axs[0, 1].set_ylabel('Chiều dài 1 (Length1)')

# Biểu đồ 3: Trọng lượng vs Chiều dài 2
axs[1, 0].scatter(df["Weight"], df["Length2"], c='yellow')
axs[1, 0].set_title('Trọng lượng vs Chiều dài 2')
axs[1, 0].set_xlabel('Trọng lượng (Weight)')
axs[1, 0].set_ylabel('Chiều dài 2 (Length2)')

# Biểu đồ 4: Chiều cao vs Chiều rộng
axs[1, 1].scatter(df["Height"], df["Width"], c='red')
axs[1, 1].set_title('Chiều cao vs Chiều rộng')
axs[1, 1].set_xlabel('Chiều cao (Height)')
axs[1, 1].set_ylabel('Chiều rộng (Width)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
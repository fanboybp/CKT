import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Houses.csv vào một DataFrame với mã hóa 'latin1'
df = pd.read_csv('E:\\Tien_Tien\\DC\\C14\\Houses.csv', encoding='latin1')
#df.drop(['id'], axis=1, inplace=True)  # Bỏ cột ID
df.drop(df.columns[1:3], axis=1, inplace=True)
# Xử lý các giá trị thiếu (NaN) nếu có
df.fillna(df.mean(), inplace=True)

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['floor', 'rooms', 'sq', 'year']]  # Đặc trưng
y = df['price']  # Giá nhà

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

# Thống kê thêm thông tin
dem = df.loc[(df['rooms'] > 5) & (df['year'] > 2000)].shape[0]
dem2 = df.loc[(df['year'] > 2000) & (df['price'] > 500000)].shape[0]

# In kết quả đánh giá và thông tin thống kê
print("Mean Absolute Error:", mae)  # Sai số tuyệt đối trung bình
print("Mean Squared Error:", mse)   # Sai số bình phương trung bình
print("Median Absolute Error:", medae) # Sai số tuyệt đối trung vị
print("Số nhà có trên 5 phòng và được xây dựng sau năm 2000:", dem)
print("Số nhà được xây dựng sau năm 2000 và có giá trên 500000:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({
    'Min': df[['floor', 'price', 'rooms', 'sq', 'year']].min(),
    'Max': df[['floor', 'price', 'rooms', 'sq', 'year']].max(),
    'Mean': df[['floor', 'price', 'rooms', 'sq', 'year']].mean()
})

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file ketqua.txt
with open('E:\\Tien_Tien\\DC\\C14\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Số nhà có trên 5 phòng và được xây dựng sau năm 2000: " + str(dem) + "\n")
    file.write("Số nhà được xây dựng sau năm 2000 và có giá trên 500000: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Biểu đồ 1: Giá nhà vs Số phòng
axs[0, 0].scatter(df["price"], df["rooms"], c='brown')
axs[0, 0].set_title('Giá nhà vs Số phòng')
axs[0, 0].set_xlabel('Giá nhà (Price)')
axs[0, 0].set_ylabel('Số phòng (Rooms)')

# Biểu đồ 2: Giá nhà vs Diện tích
axs[0, 1].scatter(df["price"], df["sq"], c='green')
axs[0, 1].set_title('Giá nhà vs Diện tích')
axs[0, 1].set_xlabel('Giá nhà (Price)')
axs[0, 1].set_ylabel('Diện tích (Square feet)')

# Biểu đồ 3: Giá nhà vs Số tầng
axs[1, 0].scatter(df["price"], df["floor"], c='yellow')
axs[1, 0].set_title('Giá nhà vs Số tầng')
axs[1, 0].set_xlabel('Giá nhà (Price)')
axs[1, 0].set_ylabel('Số tầng (Floor)')

# Biểu đồ 4: Diện tích vs Số phòng
axs[1, 1].scatter(df["sq"], df["rooms"], c='red')
axs[1, 1].set_title('Diện tích vs Số phòng')
axs[1, 1].set_xlabel('Diện tích (Square feet)')
axs[1, 1].set_ylabel('Số phòng (Rooms)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
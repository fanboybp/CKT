import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file vào một DataFrame
df = pd.read_csv('E:\\Tien_Tien\\DC\\C15\\household_power_consumption.txt', sep=';')

# In ra tên các cột để kiểm tra
print("Column names in the dataset:", df.columns)

# Kiểm tra một vài dòng đầu tiên của DataFrame để chắc chắn về định dạng dữ liệu
print(df.head())

# Thay thế dấu '?' bằng NaN
df.replace('?', np.nan, inplace=True)

# Chuyển các cột sang kiểu dữ liệu số nếu có thể
df = df.apply(pd.to_numeric, errors='coerce')

# Kiểm tra số lượng giá trị NaN trong mỗi cột
print("Missing values in each column:\n", df.isna().sum())

# Quyết định cách xử lý các giá trị NaN
# Ở đây chúng ta điền NaN bằng giá trị trung bình của mỗi cột (có thể thay đổi cách xử lý nếu cần)
df.fillna(df.mean(), inplace=True)

# Kiểm tra lại DataFrame sau khi điền NaN
print("DataFrame after filling NaN values:\n", df.head())

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
# Đảm bảo tên các cột khớp với dữ liệu của bạn
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]  # Features
y = df['Global_intensity']  # Target

# Kiểm tra kích thước của X và y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

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

print("Mean Absolute Error:", mae)  # Sai số tuyệt đối trung bình
print("Mean Squared Error:", mse)   # Sai số bình phương trung bình
print("Median Absolute Error:", medae) # Sai số tuyệt đối trung vị

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({
    'Min': df[['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].min(),
    'Max': df[['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].max(),
    'Mean': df[['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
})

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('E:\\Tien_Tien\\DC\\C15\\ketqua.txt','w',encoding='utf-8') as file:
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Tính tổng số năng lượng trong ngày 16/12/2006
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
energy_16122006_active = df.loc[df['Date'] == '2006-12-16', 'Global_active_power'].sum()
energy_16122006_reactive = df.loc[df['Date'] == '2006-12-16', 'Global_reactive_power'].sum()

print("Tổng số năng lượng (Global_active_power) trong ngày 16/12/2006:", energy_16122006_active)
print("Tổng số năng lượng (Global_reactive_power) trong ngày 16/12/2006:", energy_16122006_reactive)

# Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Global_intensity vs Sub_metering_1
axs[0, 0].scatter(df["Global_intensity"], df["Sub_metering_1"], c='brown')
axs[0, 0].set_title('Global_intensity vs Sub_metering_1')
axs[0, 0].set_xlabel('Global_intensity')
axs[0, 0].set_ylabel('Sub_metering_1')

# Biểu đồ 2: Global_intensity vs Sub_metering_2
axs[0, 1].scatter(df["Global_intensity"], df["Sub_metering_2"], c='green')
axs[0, 1].set_title('Global_intensity vs Sub_metering_2')
axs[0, 1].set_xlabel('Global_intensity')
axs[0, 1].set_ylabel('Sub_metering_2')

# Biểu đồ 3: Global_intensity vs Sub_metering_3
axs[1, 0].scatter(df["Global_intensity"], df["Sub_metering_3"], c='yellow')
axs[1, 0].set_title('Global_intensity vs Sub_metering_3')
axs[1, 0].set_xlabel('Global_intensity')
axs[1, 0].set_ylabel('Sub_metering_3')

# Biểu đồ 4: Sub_metering_3 vs Sub_metering_2
axs[1, 1].scatter(df["Sub_metering_3"], df["Sub_metering_2"], c='red')
axs[1, 1].set_title('Sub_metering_3 vs Sub_metering_2')
axs[1, 1].set_xlabel('Sub_metering_3')
axs[1, 1].set_ylabel('Sub_metering_2')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
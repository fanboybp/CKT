import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Abalone.data vào một DataFrame
df=pd.read_csv('E:\\Tien_Tien\\DC\\C2\\Real estate valuation data set.csv')
df.drop(['No'], axis=1, inplace=True) # bỏ cột No
# Bước 2: Chuẩn bị dữ liệu cho việc uấn luyện mô hình
X = df[['X1 transaction date','X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area']]  # Đặc trưng
y = df['X2 house age']  # tuổi

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
#lấy giá trị  của ng 30 tuổi có trên lương trên 50k$
dem = df.loc[(df['X2 house age'] > 30) & (df['Y house price of unit area'] > 50)].shape[0]
dem2 = df.loc[(df['Y house price of unit area'] < 30) & (df['X3 distance to the nearest MRT station'] < 1)].shape[0]
print("Mean Absolute Error:", mae)  #sai số tuyệt đốitrung bình
print("Mean Squared Error:", mse)   #Sai số bình phương trung bình
print("Median Absolute Error:", medae) #Sai số tuyệt đối trung vị
#lấy giá trị  của ng 30 tuổi có trên lương trên 50k$
print("Nhà trên 30 tuổi mà có giá lơn hơn 50  :"+str(dem))
print("Nhà có giá dưới 30 mà cách ga tàu 1km :"+str(dem2))

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': df.min(), 'Max': df.max(), 'Mean': df.mean()}, index=['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores','Y house price of unit area'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('E:\\Tien_Tien\\DC\\C2\\ketqua.txt','w',encoding='utf-8') as file:
    file.write("Nhà trên 30 tuổi mà có giá lơn hơn 50: " + str(dem) + "\n")
    file.write("Nhà có giá dưới 30 mà cách ga tàu 1km: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tuổi vs Kinh nhiệm
axs[0, 0].scatter(df["X2 house age"], df["Y house price of unit area"], c='brown')
axs[0, 0].set_title('Tuổi vs Giá nhà')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('Giá (Price)')

# Biểu đồ 2: Giá nhà với khoảng cách ga tàu
axs[0, 1].scatter(df["Y house price of unit area"], df["X3 distance to the nearest MRT station"], c='green')
axs[0, 1].set_title('Giá nhà với khoảng cách ga tàu')
axs[0, 1].set_xlabel('Giá nhà (Price)')
axs[0, 1].set_ylabel('Khoảng cách tới ga tàu')

# Biểu đồ 3: Giá nhà với số cửa hàng tiện lợi
axs[1, 0].scatter(df["Y house price of unit area"], df["X4 number of convenience stores"], c='yellow')
axs[1, 0].set_title('Giá nhà với số cửa hàng tiện lợi')
axs[1, 0].set_xlabel('Giá nhà ')
axs[1, 0].set_ylabel('Số cửa hàng tiện lợi')

# Biểu đồ 4: Số cửa hàng tiện lợi với khoảng cách ga tàu
axs[1, 1].scatter(df["X4 number of convenience stores"], df["X3 distance to the nearest MRT station"], c='red')
axs[1, 1].set_title('Số cửa hàng tiện lợi với khoảng cách ga tàu')
axs[1, 1].set_xlabel('Số của hàng tiện lợi')
axs[1, 1].set_ylabel('Khoảng cách tới ga tàu')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
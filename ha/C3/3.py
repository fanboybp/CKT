import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Abalone.data vào một DataFrame
df=pd.read_csv('E:\\Tien_Tien\\DC\\C3\\Boston-house-price-data.csv')
# Bước 2: Chuẩn bị dữ liệu cho việc uấn luyện mô hình
X = df[['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']]  # Đặc trưng
y = df['AGE']  # tuổi

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
dem = df.loc[(df['AGE'] > 30) & (df['MEDV'] > 30)].shape[0]
dem2 = df.loc[(df['MEDV'] < 30) & (df['CRIM'] < 0.1)].shape[0]
print("Mean Absolute Error:", mae)  #sai số tuyệt đốitrung bình
print("Mean Squared Error:", mse)   #Sai số bình phương trung bình
print("Median Absolute Error:", medae) #Sai số tuyệt đối trung vị
#lấy giá trị  của ng 30 tuổi có trên lương trên 50k$
print("Nhà trên 30 tuổi mà có giá lơn hơn 50  :"+str(dem))
print("Nhà có giá dưới 30 mà cách ga tàu 1km :"+str(dem2))

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': df.min(), 'Max': df.max(), 'Mean': df.mean()}, index=['CRIM', 'INDUS', 'CHAS','RM','AGE','MEDV'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('E:\\Tien_Tien\\DC\\C3\\ketqua.txt','w',encoding='utf-8') as file:
    file.write("Ngôi nhà mà có tuổi trên 30 mà có giá trên 30: " + str(dem) + "\n")
    file.write("Ngôi nhà mà có giá dưới 30 và tỷ lệ tội phạm dưới 0.1: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Giá nhà TB 1000$ vói tỷ lệ tội phạm bình quân đầu người
axs[0, 0].scatter(df["MEDV"], df["CRIM"], c='brown')
axs[0, 0].set_title('Giá nhà TB 1000$ vs Tỷ lệ tội phạm bình quân')
axs[0, 0].set_xlabel('Giá nhà TB (MEDV)')
axs[0, 0].set_ylabel('Tỷ lệ tội phạm (CRIM)')

# Biểu đồ 2: Giá nhà TB 1000$ vs số phòng TB mỗi nhà
axs[0, 1].scatter(df["MEDV"], df["RM"], c='green')
axs[0, 1].set_title('Giá nhà TB 1000$ vs số phòng TB mỗi nhà')
axs[0, 1].set_xlabel('Giá nhà TB 1000$ (MEDV)')
axs[0, 1].set_ylabel('Số phòng TB mỗi nhà (RM)')

# Biểu đồ 3: Giá nhà TB 1000$ vs Tuổi
axs[1, 0].scatter(df["MEDV"], df["AGE"], c='yellow')
axs[1, 0].set_title('Giá nhà TB 1000$ vs Tuổi')
axs[1, 0].set_xlabel('Giá nhà TB 1000$ (MEDV)')
axs[1, 0].set_ylabel('Tuổi (AGE)')

# Biểu đồ 4: Tuổi vs Số phòng TB mỗi nhà
axs[1, 1].scatter(df["AGE"], df["RM"], c='red')
axs[1, 1].set_title('Tuổi vs Số phòng TB mỗi nhà')
axs[1, 1].set_xlabel('Tuổi (AGE)')
axs[1, 1].set_ylabel('Số phòng TB mỗi nhà (RM)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
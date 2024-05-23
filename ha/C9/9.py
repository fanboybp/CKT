import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file Abalone.data vào một DataFrame
df=pd.read_csv('E:\\Tien_Tien\\DC\\C9\\Food demand.csv')
df.drop(['id'], axis=1, inplace=True) # bỏ cột ID
# Bước 2: Chuẩn bị dữ liệu cho việc uấn luyện mô hình
X = df[['week', 'center_id', 'meal_id', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']]  # Đặc trưng
y = df['checkout_price']  # tuổi

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
# Lấy số trung tâm có số lượng đơn > 200

print("Mean Absolute Error:", mae)  #sai số tuyệt đốitrung bình
print("Mean Squared Error:", mse)   #Sai số bình phương trung bình
print("Median Absolute Error:", medae) #Sai số tuyệt đối trung vị
dem = df.loc[df['num_orders'] > 200, 'center_id'].nunique()

# Lấy loại thực phẩm tiêu thụ nhiều nhất
dem2 = df.groupby('meal_id')['num_orders'].sum().idxmax()

print("Số trung tâm có số lượng đơn > 200:", dem)
print("Loại thực phẩm tiêu thụ nhiều nhất:", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = pd.DataFrame({'Min': df.min(), 'Max': df.max(), 'Mean': df.mean()}, index=['checkout_price', 'base_price', 'emailer_for_promotion','homepage_featured','num_orders'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)
with open('E:\\Tien_Tien\\DC\\C9\\ketqua.txt','w',encoding='utf-8') as file:
    file.write("Số trung tâm có số lượng đơn > 200: " + str(dem) + "\n")
    file.write("Loại thực phẩm tiêu thụ nhiều nhất: " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Giá thanh toán với giá cơ sở
axs[0, 0].scatter(df["checkout_price"], df["base_price"], c='brown')
axs[0, 0].set_title('Giá thanh toán với giá cơ sở')
axs[0, 0].set_xlabel('Giá thanh toán (checkout_price)')
axs[0, 0].set_ylabel('Giá cơ sở (base_price)')

# Biểu đồ 2: Giá thanh toán với số thứ tự
axs[0, 1].scatter(df["checkout_price"], df["num_orders"], c='green')
axs[0, 1].set_title('Giá thanh toán với số thứ tự')
axs[0, 1].set_xlabel('Giá thanh toán (checkout_price)')
axs[0, 1].set_ylabel('Số thứ tự (num_orders)')

# Biểu đồ 3: Giá thanh toán với trang chủ
axs[1, 0].scatter(df["checkout_price"], df["homepage_featured"], c='yellow')
axs[1, 0].set_title('Giá thanh toán với trang chủ')
axs[1, 0].set_xlabel('Giá thanh toán (checkout_price)')
axs[1, 0].set_ylabel('Trang chủ (homepage_featured)')

# Biểu đồ 4: Giá cơ sở với người gửi Email
axs[1, 1].scatter(df["base_price"], df["emailer_for_promotion"], c='red')
axs[1, 1].set_title('Giá cơ sở với người gửi Email')
axs[1, 1].set_xlabel('Giá cơ sở (base_price)')
axs[1, 1].set_ylabel('Người gửi Email (emailer_for_promotion)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
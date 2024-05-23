import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV và bỏ cột 'GallusID'
df = pd.read_csv('E:\\Tien_Tien\\DC\\C5\\GallusGallusDomesticus.csv')
df.drop(['GallusID'], axis=1, inplace=True)

# Bước 2: Chuẩn bị dữ liệu cho việc huấn luyện mô hình
X = df[['GallusBreed', 'Day', 'GallusWeight', 'GallusEggColor', 'GallusEggWeight', 'AmountOfFeed', 'EggsPerDay', 'GallusCombType', 'SunLightExposure', 'GallusClass', 'GallusLegShanksColor', 'GallusBeakColor', 'GallusEarLobesColor', 'GallusPlumage']]
y = df['Age']  # Nhãn (tuổi)

# Sử dụng One-Hot Encoding để chuyển đổi các đặc trưng phân loại thành các đặc trưng số
X = pd.get_dummies(X, columns=['GallusBreed', 'GallusEggColor', 'GallusCombType', 'GallusClass', 'GallusLegShanksColor', 'GallusBeakColor', 'GallusEarLobesColor', 'GallusPlumage'])

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

# Lấy giá trị của gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown'
dem = df.loc[(df['GallusEggColor'] == 'Brown') & (df['GallusWeight'] > 3000)].shape[0]
# Lấy giá trị của gà có tuổi lớn hơn 800 và màu trứng là 'Brown'
dem2 = df.loc[(df['GallusEggColor'] == 'Brown') & (df['Age'] > 800)].shape[0]

print("Gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown':", dem)
print("Gà có tuổi lớn hơn 800 và màu trứng là 'Brown':", dem2)

# Tạo DataFrame từ các giá trị tối thiểu, tối đa và trung bình
summary_table = df[['Age', 'GallusWeight', 'GallusEggWeight', 'EggsPerDay', 'AmountOfFeed']].agg(['min', 'max', 'mean'])

# Hiển thị bảng với các thuộc tính nằm theo hàng ngang (chuyển vị)
summary_table_transposed = summary_table.transpose()
print(summary_table_transposed)

# Ghi kết quả vào file
with open('E:\\Tien_Tien\\DC\\C5\\ketqua.txt', 'w', encoding='utf-8') as file:
    file.write("Gà có trọng lượng lớn hơn 3000 và màu trứng là 'Brown': " + str(dem) + "\n")
    file.write("Gà có tuổi lớn hơn 800 và màu trứng là 'Brown': " + str(dem2) + "\n\n")
    file.write("Bảng tóm tắt:\n")
    file.write(summary_table_transposed.to_string())

# Bước 5: Biểu diễn các tương quan bằng biểu đồ
# Tạo khung hình và các subplot
fig, axs = plt.subplots(2, 2, figsize=(15, 7))

# Biểu đồ 1: Tuổi vs Trọng lượng của trứng
axs[0, 0].scatter(df["Age"], df["GallusEggWeight"], c='brown')
axs[0, 0].set_title('Tuổi vs Trọng lượng của trứng')
axs[0, 0].set_xlabel('Tuổi (Age)')
axs[0, 0].set_ylabel('Trọng lượng của trứng (GallusEggWeight)')

# Biểu đồ 2: Tuổi với trọng lượng của gà
axs[0, 1].scatter(df["Age"], df["GallusWeight"], c='green')
axs[0, 1].set_title('Tuổi với trọng lượng của gà')
axs[0, 1].set_xlabel('Tuổi (Age)')
axs[0, 1].set_ylabel('Trọng lượng của gà (GallusWeight)')

# Biểu đồ 3: Tuổi với Số trứng đẻ trong 1 ngày
axs[1, 0].scatter(df["Age"], df["EggsPerDay"], c='yellow')
axs[1, 0].set_title('Tuổi với Số trứng đẻ trong 1 ngày')
axs[1, 0].set_xlabel('Tuổi (Age) ')
axs[1, 0].set_ylabel('Số trứng đẻ trong 1 ngày (EggsPerDay)')

# Biểu đồ 4: Trọng lượng của trứng vs Lượng thức ăn tiêu thụ mỗi ngày
axs[1, 1].scatter(df["GallusEggWeight"], df["AmountOfFeed"], c='red')
axs[1, 1].set_title('Trọng lượng của trứng vs Lượng thức ăn tiêu thụ mỗi ngày')
axs[1, 1].set_xlabel('Trọng lượng của trứng (GallusEggWeight)')
axs[1, 1].set_ylabel('Lượng thức ăn tiêu thụ mỗi ngày(AmountOfFeed)')

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
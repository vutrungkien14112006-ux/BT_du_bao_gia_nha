import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('nha_dat.csv')

# 2. Tách đặc trưng (X) và mục tiêu (y)
X = df[['Dien_tich', 'Phong_ngu', 'Tuoi_nha']]
y = df['Gia_ban']

# 3. Gọi mô hình Hồi quy tuyến tính và cho học dữ liệu
model = LinearRegression()
model.fit(X, y)

# 4. Dự đoán thử một căn nhà mới: 70m2, 2 phòng ngủ, đã xây 4 năm
nha_moi = pd.DataFrame({'Dien_tich': [70], 'Phong_ngu': [2], 'Tuoi_nha': [4]})
gia_du_doan = model.predict(nha_moi)

print(f"Giá dự báo cho căn nhà này là: {gia_du_doan[0]:.2f} Tỷ VNĐ")
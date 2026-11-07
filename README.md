# Drought-Prediction

Powered by Copilot

Clone nhánh lstm để có thể so sánh mô hình, các file được lưu vào đúng thư mục, nếu có lỗi nói Châu để Châu làm lại nhé
Công thức sử dụng là công thức
Trong notebook XGBoost, mô hình được huấn luyện để dự báo SPEI-3 của tháng kế tiếp tại từng ô lưới.

Tạo nhãn: với ô lưới (g) và tháng (t), đầu vào là vector đặc trưng (X_{g,t}), còn nhãn là:

[
y_{g,t}=\mathrm{SPEI3}_{g,t+1}
]

Chỉ giữ các cặp tháng liên tiếp. SPEI-3 đã có sẵn trong dữ liệu; notebook không tự tính chỉ số SPEI.

Đặc trưng: nhiệt độ, lượng mưa, cân bằng nước, độ ẩm đất, SPI-3, SPEI-3 hiện tại, các biến trễ, tọa độ và mùa vụ. Mùa vụ được mã hóa theo tháng (m):

[
\text{month_sin}=\sin(2\pi m/12),\qquad
\text{month_cos}=\cos(2\pi m/12)
]

Công thức dự báo XGBoost: mô hình cộng dự đoán từ nhiều cây:

[
\hat y_{g,t}=\sum_{k=1}^{K} f_k(X_{g,t})
]

Các cây được tối ưu theo sai số bình phương, có thêm hạng phạt độ phức tạp của cây:

[
\mathcal{L}=\sum_i(y_i-\hat y_i)^2+\sum_k\Omega(f_k)
]

*Xin lỗi Trang nhe cái github nó không hiển thị được phương trình toán viết bằng latex mà nó chỉ ra code thôi, nếu mà Trang dùng để note vào trong quyển báo cáo thì dùng các tool Latex Editor trên mạng để dịch nó thành phương trình toán rồi copy vào nhé

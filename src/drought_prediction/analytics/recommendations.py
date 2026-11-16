COASTAL_PROVINCES = {
    "Cà Mau", "Bạc Liêu", "Sóc Trăng", "Trà Vinh", "Bến Tre", "Tiền Giang", "Kiên Giang"
}

PROVINCE_DISPLAY_MAP = {
    "AnGiang": "An Giang", "BạcLiêu": "Bạc Liêu", "BếnTre": "Bến Tre",
    "CàMau": "Cà Mau", "CầnThơ": "Cần Thơ", "ĐồngTháp": "Đồng Tháp",
    "HậuGiang": "Hậu Giang", "KiênGiang": "Kiên Giang", "LongAn": "Long An",
    "SócTrăng": "Sóc Trăng", "TiềnGiang": "Tiền Giang", "TràVinh": "Trà Vinh",
    "VĩnhLong": "Vĩnh Long",
}

DISCLAIMER = "*(Thông tin tham khảo hỗ trợ quyết định, không thay thế văn bản chỉ đạo nông nghiệp chính thức)*"

def format_province_name(raw_name: str) -> str:
    if not raw_name or str(raw_name) == "nan":
        return "Ngoài ranh giới ĐBSCL"
    return PROVINCE_DISPLAY_MAP.get(str(raw_name), str(raw_name))

def get_agricultural_recommendation(severity: str, province: str = "") -> str:
    display_prov = format_province_name(province)
    is_coastal = display_prov in COASTAL_PROVINCES

    if severity == "Bình thường":
        base = "Duy trì lịch canh tác và chế độ tưới tiêu định kỳ; tiếp tục theo dõi bản tin khí tượng thủy văn."
    elif severity == "Hạn nhẹ":
        base = (
            "Theo dõi chặt chẽ độ mặn tại các cửa sông; tranh thủ lấy nước ngọt vào ao chứa/kênh nội đồng khi triều thấp."
            if is_coastal else
            "Giám sát độ ẩm đất tầng rễ; áp dụng kỹ thuật tưới ngập khô xen kẽ (AWD) để tiết kiệm nước."
        )
    elif severity == "Hạn vừa":
        base = (
            "Đóng kín các cống ngăn mặn đầu mối; hạn chế xuống giống lúa vụ mới tại vùng ven biển; ưu tiên cấp nước sinh hoạt và vườn cây ăn trái."
            if is_coastal else
            "Ưu tiên nguồn nước tưới cho diện tích lúa đang giai đoạn đòng - trỗ; nạo vét kênh mương nội đồng và điều tiết bơm luân phiên."
        )
    elif severity == "Hạn nặng đến cực đoan":
        base = (
            "BÁO ĐỘNG ĐỎ: Kích hoạt phương án phòng chống hạn mặn khẩn cấp; dừng xuống giống hoàn toàn ở vùng thiếu nước; sử dụng giống chịu mặn (ST24, ST25)."
            if is_coastal else
            "BÁO ĐỘNG ĐỎ: Tạm ngưng mở rộng diện tích xuống giống; huy động trạm bơm dã chiến tận dụng nguồn nước ngọt còn lại; chuyển đổi cơ cấu cây trồng cạn ngắn ngày."
        )
    else:
        base = "Chưa đủ dữ liệu để đưa ra khuyến nghị cụ thể."
    return f"{base} {DISCLAIMER}"

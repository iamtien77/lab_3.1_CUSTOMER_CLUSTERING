# THIẾT KẾ THUẬT TOÁN K-MEANS

## 1. Purpose - Mục đích

- Nhóm dùng K-Means làm baseline cho bài lab phân cụm khách hàng.
- Nhóm tìm hiểu cách gom các khách hàng tương đồng dựa trên khoảng cách Euclidean.
- Nhóm khảo sát ảnh hưởng của số cụm K và giải thích đặc điểm từng cụm.
- Thiết kế cụ thể hóa mục 6.5 trong [thiết kế tổng thể](design.md).

## 2. Data used - Dữ liệu sử dụng

- Nguồn raw nội bộ là `data/raw/marketing_campaign.csv`; nguồn tải gốc cần xác minh.
- Theo thiết kế tổng thể, nhóm dự kiến phân cụm bằng tuổi, thu nhập và điểm chi tiêu.
- Dữ liệu raw có `Year_Birth`, `Income` và các khoản chi tiêu; không có sẵn `Age`, `Gender` hoặc `Spending Score`.
- Nếu tính tuổi từ năm sinh, nhóm cần thống nhất năm tham chiếu phù hợp với thời điểm dữ liệu.
- Nhóm không tự xem tổng các khoản chi tiêu là `Spending Score` hay tự tạo giới tính.
- Trước khi thử nghiệm, nhóm cần thống nhất tập dữ liệu đáp ứng đề hoặc cách điều chỉnh đặc trưng và thiết kế tổng thể.
- `ID` chỉ phục vụ đối chiếu bản ghi; các cột phản hồi chiến dịch không được coi là nhãn phân khúc đúng.
- Nhóm dự kiến phân tích toàn bộ tập dữ liệu cho mục tiêu khám phá của bài lab.

## 3. Input - Đầu vào

- Ma trận `X_scaled` có n khách hàng và p thuộc tính số đã được nhóm thống nhất.
- Với phương án theo đề, p = 3 tương ứng tuổi, thu nhập và điểm chi tiêu.
- Ma trận cần chứa giá trị hữu hạn, không còn giá trị thiếu và giữ đúng thứ tự khách hàng.
- Các thuộc tính được chuẩn hóa theo [thiết kế Feature Scaling](feature_scaling_design.md).
- Bảng dữ liệu theo đơn vị gốc và mã đối chiếu được giữ riêng để diễn giải cụm.
- Nhóm cần có ít nhất ba khách hàng và hai điểm phân biệt để cân nhắc baseline K = 2 và tính Silhouette hợp lệ.

## 4. Output - Đầu ra dự kiến

- Một nhãn cụm cho mỗi khách hàng, gắn đúng với mã đối chiếu và dòng dữ liệu.
- K tâm cụm trong không gian chuẩn hóa, kèm cách quy đổi về đơn vị đặc trưng khi cần.
- Thông tin cấu hình thử nghiệm, số cụm thực tế, Inertia và trạng thái hội tụ.
- Bảng quy mô cụm và thống kê đặc trưng phục vụ mô tả khách hàng.
- Thông tin đầu ra là yêu cầu thiết kế, không ấn định trước cấu trúc phân khúc.

## 5. Tham số dự kiến

| Thành phần | Lựa chọn ban đầu | Mục đích |
|---|---|---|
| Baseline | K = 2 nếu dữ liệu đủ điều kiện | Tạo mốc đối chiếu đơn giản |
| Khảo sát K | Từ 2 đến 10, giới hạn bởi n - 1 và số điểm phân biệt | Khảo sát mức độ phân chia |
| Khởi tạo | k-means++ | Hỗ trợ lựa chọn tâm ban đầu |
| Số lần khởi tạo | 10 | Giảm phụ thuộc vào một lần khởi tạo |
| Random seed | 42 | Giúp lặp lại thử nghiệm cùng cấu hình |
| Giới hạn vòng lặp | 300 | Giới hạn thời gian cho mỗi lần khởi tạo |

Các giá trị trên là đề xuất cho bài lab; nhóm giữ chúng cố định khi khảo sát riêng K.

## 6. Trình tự thực hiện dự kiến

1. Nhóm xác nhận đặc trưng, điều kiện dữ liệu và cách chuẩn hóa.
2. Nhóm thiết lập baseline K = 2 với các tham số đã thống nhất.
3. Với mỗi lần khởi tạo, thuật toán chọn K tâm và gán từng điểm vào tâm gần nhất.
4. Thuật toán cập nhật mỗi tâm bằng trung bình các điểm thuộc cụm đó.
5. Thuật toán lặp bước gán và cập nhật tới khi hội tụ hoặc đạt giới hạn vòng lặp.
6. Nhóm dùng lần khởi tạo có Inertia nhỏ nhất trong cùng cấu hình K.
7. Nhóm lặp lại với các K hợp lệ, giữ cùng dữ liệu và phép chuẩn hóa.
8. Nhóm đối chiếu các tiêu chí đánh giá trước khi lựa chọn cấu hình để diễn giải.

## 7. Evaluation metrics - Tiêu chí đánh giá

- Silhouette coefficient là chỉ số chính, được tính trên `X_scaled` với khoảng cách Euclidean.
- Nhóm chỉ tính Silhouette khi số nhãn thực tế nằm trong khoảng từ 2 đến n - 1.
- Nếu điều kiện không thỏa mãn, nhóm ghi trạng thái không đủ điều kiện thay vì gán điểm giả.
- Silhouette cao hơn thường thể hiện sự gắn kết và tách biệt thuận lợi hơn, nhưng không tự chứng minh phân khúc có ý nghĩa.
- Nhóm dùng Inertia để vẽ Elbow; chỉ đối chiếu trên cùng mẫu, đặc trưng và thang đo.
- Nhóm kết hợp quy mô cụm, thống kê theo đơn vị gốc và Scatter Plot khi xem xét K.
- Nhóm không dùng Accuracy hoặc chọn K chỉ vì Inertia giảm; không cam kết trước ngưỡng điểm.

## 8. Hạn chế và điều kiện cần kiểm tra

- K-Means có thể chịu ảnh hưởng của ngoại lai, thang đo và vị trí tâm khởi tạo.
- Cụm kéo dài, mật độ khác nhau hoặc chồng lấp có thể khó được mô tả bằng các tâm.
- Elbow có thể không rõ; nhóm cần giải thích lựa chọn bằng nhiều căn cứ.
- Nếu số cụm thực tế nhỏ hơn K hoặc thuật toán chưa hội tụ, nhóm cần xem lại cấu hình trước khi diễn giải.
- Nhãn cụm không thể hiện thứ hạng khách hàng; nhóm không suy diễn đặc điểm nhạy cảm từ nhãn.
- Nhóm ưu tiên thống kê tổng hợp khi trình bày, giữ mã khách hàng để đối chiếu nội bộ.

## 9. Status - Trạng thái thiết kế

- Vai trò: phương án baseline chính trong kế hoạch bài lab.
- Trạng thái: dự kiến thực hiện, chờ thống nhất dữ liệu và đặc trưng phù hợp với đề.
- Điều kiện chuyển sang thử nghiệm: thống nhất đặc trưng, hoàn tất kiểm tra dữ liệu và chuẩn hóa, xác nhận tiêu chí đánh giá.
- Tài liệu mô tả kế hoạch thuật toán; không xác nhận việc triển khai hoặc chất lượng thực nghiệm.

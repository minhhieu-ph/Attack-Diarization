# Attack Diarization cho NIDS dựa trên NetFlow

Mã nguồn cho đề cương "Attack Diarization cho NIDS". Bài toán: từ các flow NetFlow có timestamp,
xác định các **đợt tấn công**, gồm ai tấn công ai, kiểu gì, từ lúc nào đến lúc nào. Code đi theo
đúng 6 câu hỏi của đề cương: dữ liệu (mục 3), giả thiết H1–H5 (mục 4) và thực nghiệm E1–E8 (mục 5).

## Cấu trúc

```
attack-diarization/
├── dataset/                  # dữ liệu (xem dataset/README.md)
│   ├── raw/  processed/  interim/
│   └── schedules/            # lịch tấn công = ground truth đợt cho NetFlow v3
├── configs/
│   ├── default.yaml          # mọi tham số
│   ├── b300.yaml             # cho job 1×B300, 28 CPU, 450 GiB RAM
│   ├── datasets.yaml         # nguồn tải, loader, nhóm tấn công cho thử nghiệm chéo bộ
│   └── smoke.yaml            # chạy thử trên dữ liệu giả lập
├── src/attackdiar/
│   ├── data/                 # tải, đọc, ground truth đợt, split, host-bin, kiểm tra áp lực
│   ├── metrics/              # F1 theo đợt, quy kết, EER, TTD, FA/giờ, kiểm định, phân loại lỗi
│   ├── baselines/            # B1–B7: XGBoost, FT-Transformer, GNN, TGN-lite + hậu xử lý
│   ├── models/               # FlowDiar + giải mã
│   ├── pipeline.py  experiments.py  report.py  feasibility.py
├── scripts/                  # feasibility, download, prepare, run, check_env, run_gate.sh, run_all.sh, smoke.sh
├── tests/                    # unit test + end-to-end trên CPU
├── environment.yml  setup_env.sh  Makefile  pyproject.toml
└── outputs/                  # kết quả, mô hình, dự đoán, report.md
```

## Cài đặt

```bash
bash setup_env.sh            # tạo env conda "attackdiar", cài torch khớp CUDA, chạy check_env
conda activate attackdiar
make test                    # unit test (CPU)
make smoke                   # chạy toàn bộ E1–E8 trên dữ liệu giả lập, vài phút
```

`setup_env.sh` đọc phiên bản CUDA từ `nvidia-smi`: CUDA ≥ 13 thì cài wheel `cu130`, CUDA 12.8–12.9
thì cài `cu128`. B300 (Blackwell) cần CUDA ≥ 12.8. Muốn ép một wheel cụ thể:
`TORCH_CUDA=cu128 bash setup_env.sh`. `check_env.py` chạy thử một phép nhân bf16 trên GPU và XGBoost
trên GPU. Nếu torch không thấy GPU dù `nvidia-smi` vẫn chạy, thì driver không hỗ trợ CUDA của wheel;
khi đó cài lại với `cu128`.

## Dữ liệu

```bash
python scripts/download.py --config configs/b300.yaml     # CIDDS-001 tải thẳng; NetFlow v3 thử tải tự động
python scripts/prepare.py  --config configs/b300.yaml     # chuẩn hóa, ground truth đợt, split
```

- Link UQ của NetFlow v3 là trang trung gian. Nếu tải tự động không được, script sẽ in hướng dẫn:
  tải zip bằng trình duyệt, chép vào `dataset/raw/<tên>/`, rồi chạy lại `prepare.py`.
- **Ground truth đợt phải độc lập với baseline.** CIDDS-001 dùng `attackID` có sẵn. Với bốn bộ
  NetFlow v3, cần điền `dataset/schedules/<tên>.csv` (có file mẫu `.template`, định dạng ở
  `dataset/README.md`). Nếu thiếu lịch, code vẫn chạy bằng heuristic nhưng ghi
  `gt_independent: false`, và cổng H1 sẽ báo INVALID cho bộ đó.

## Thử khả thi trước (pilot)

Chạy bước này trước khi đầu tư vào FlowDiar. Nó trả lời câu hỏi: baseline mạnh còn để lại
khoảng trống ở mức đợt hay không, và khoảng trống đó đến từ đâu.

```bash
python scripts/feasibility.py --config configs/b300.yaml              # mặc định: CIDDS-001
python scripts/feasibility.py --config configs/b300.yaml --flowdiar   # thêm FlowDiar nhanh (8 epoch)
python scripts/feasibility.py --config configs/b300.yaml --set data.max_rows=5000000  # chạy thử nhanh hơn
```

Kết quả ghi vào `outputs/feasibility/summary.md`, gồm các mục:
C1 ground truth có độc lập không, C2 cỡ mẫu, C3 tỉ lệ chồng lấn, C4 số kẻ tấn công,
**C5 trần oracle** (nhãn flow đúng + gom theo khoảng hở τ), C6 B1/B2/B7 trên split thời gian,
C7 B1 trên split tách host, C8 phân loại lỗi, C9 FlowDiar (tùy chọn).
Kết luận cuối là GO / STOP / CHƯA RÕ theo ngưỡng cổng trong `configs/default.yaml`.

Cách đọc C5: nếu oracle vẫn dưới 0.95 thì cách gom theo thời gian sai ngay cả khi phân loại flow
hoàn hảo, tức bài toán gom đợt tồn tại độc lập. Đây là luận điểm mạnh nhất cho hướng này.
Nếu oracle gần 1 thì khoảng trống (nếu có) chỉ đến từ lỗi phân loại flow.

Mặc định chỉ chạy CIDDS-001 vì bộ này tải thẳng được và có ground truth độc lập. Bốn bộ NetFlow v3
cần lịch tấn công; nếu thiếu, C1 sẽ là FAIL.

## Chạy thực nghiệm

| Lệnh | Thực nghiệm | Trả lời |
|---|---|---|
| `python scripts/run.py --exp E1` | Thống kê ground truth, độ khớp với heuristic τ | tiền đề |
| `--exp E2` | B1–B7 trên split theo thời gian, 5 seed | H1 |
| `--exp E3` | Phân loại lỗi của baseline tốt nhất | H1, thiết kế |
| `--exp E4` | FlowDiar trên split theo thời gian | H2 |
| `--exp E5` | Split tách host + chéo bộ dữ liệu | H3 |
| `--exp E6` | Ablation: bỏ đồ thị, bỏ thời gian, đổi bộ giải mã | H4 |
| `--exp E7` | Chậm rãi, xen benign, chia nhiều IP nguồn | H5 |
| `--exp E8` | Trực tuyến: time-to-detect | H2 |
| `--exp report` | Gom bảng, quyết định cổng H1, kiểm định H2 | |

```bash
bash scripts/run_gate.sh --config configs/b300.yaml   # tuần 1: tải, chuẩn bị, E1–E3, quyết định cổng
bash scripts/run_all.sh  --config configs/b300.yaml   # E4–E8 chỉ chạy khi cổng là GO (FORCE=1 để bỏ qua)
python scripts/run.py --config configs/b300.yaml --exp E2 --models B1 B7 --datasets cidds001
python scripts/run.py --config configs/b300.yaml --exp E4 --set seeds=[0] --set flowdiar.epochs=10
```

Mỗi thực nghiệm ghi thêm dòng vào `outputs/results/<EXP>.jsonl` và bỏ qua những tổ hợp
(bộ dữ liệu, split, mô hình, seed) đã chạy, nên có thể dừng giữa chừng rồi chạy tiếp.
`outputs/report.md` có quyết định cổng, bảng trung bình ± độ lệch chuẩn và kết quả kiểm định.

## Thiết kế chính

- **Ground truth và B1 dùng chung một hàm gom đợt** (`data/episodes.py`). Vì vậy khác biệt giữa
  chúng chỉ đến từ nhãn flow, không đến từ code.
- **Chỉ số theo đợt** (`metrics/episode.py`): ghép một-một bằng thuật toán Hungary trên IoU thời gian
  (có collar) trong từng loại, nên một đợt dự đoán dài chỉ được tính trúng một đợt thật. Ngoài ra có
  quy kết A/V (Jaccard), EER kiểu DER, TTD, FA/giờ lưu lượng benign và tỉ lệ nén cảnh báo.
- **FlowDiar** (`models/flowdiar.py`) làm việc trên lưới host × time-bin, nên chi phí không phụ thuộc
  số flow (đợt DDoS hàng triệu flow vẫn rẻ). Mô hình gồm Transformer theo thời gian cho từng host,
  truyền thông điệp có hướng trên đồ thị host của cửa sổ, và hai đầu ra loại/vai trò. Giải mã bằng
  ngưỡng trễ thành đợt kẻ tấn công, sau đó gộp thành đợt nhiều kẻ tấn công.
- **Chế độ trực tuyến**: mỗi bin được phát ra bởi cửa sổ sớm nhất chứa nó trong `stride` bin cuối,
  thời điểm cảnh báo là lúc cửa sổ đó kết thúc, nên không dùng dữ liệu tương lai.
- **Tune cùng ngân sách** (`tuning.n_trials`) cho mọi mô hình. Tham số hậu xử lý được chọn trên tập val.

## Những gì đã kiểm tra

Unit test và test end-to-end trên CPU (dữ liệu, ground truth, split, B1/B2/B3/B7, chỉ số, E1–E3,
kiểm tra áp lực, report) đều qua. Các phần PyTorch (B4–B6, FlowDiar, E4–E8) chưa được chạy trong
môi trường viết code vì môi trường đó không cài được PyTorch. Hãy chạy `make smoke` trên máy GPU
trước khi chạy dữ liệu thật.

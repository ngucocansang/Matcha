import mujoco
import mujoco.viewer
import time

# 1. Load model từ file XML
model = mujoco.MjModel.from_xml_path('robot.xml')
data = mujoco.MjData(model)

# 2. Mở cửa sổ Viewer
# Viewer này cho phép bạn dùng chuột để kéo/thả robot và xem các lực tác động
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Thiết lập thời gian bắt đầu
    start_time = time.time()

    print("Đang chạy mô phỏng... Nhấn Ctrl+C để dừng.")
    
    while viewer.is_running():
        step_start = time.time()

        # Thực hiện một bước mô phỏng vật lý
        mujoco.mj_step(model, data)

        # Cập nhật hình ảnh hiển thị (đồng bộ theo thời gian thực)
        viewer.sync()

        # Điều khiển tốc độ mô phỏng để khớp với thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
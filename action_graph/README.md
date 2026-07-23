# Isaac Sim Action Graph 邊界

此目錄保留給未來 Isaac Sim 5.1 ROS 2 Bridge / Action Graph asset 與設定文件。

Action Graph 將在 Isaac Sim 與外部 ROS 2 process 之間傳輸 command、模擬
sensor data 與 robot state。Isaac Sim Python 不會內嵌 `rclpy`，也不會使用
UDP 作為最終 transport。

Flat Locomotion v1 baseline 階段尚未實作 runtime graph。

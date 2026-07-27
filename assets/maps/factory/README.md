# Factory 地圖

Deployment 入口：

`/home/ros/anymal_locomotion/assets/maps/factory/Factory_Layout.usd`

地圖與 OpenUSD 相依資產從舊 workspace 的 `anymal_d_sim_bringup/maps`
複製而來，複製過程未修改舊 workspace。`Factory_Layout.usda` 是可審查的
source layer；`Factory_Layout.usd` 是由它產生的 runtime crate。

舊 root layer 內含指回舊 workspace 的絕對路徑，現在已改成 project-relative
asset path。以下相容層：

`OpenUSD_Lab-Assets/OpenUSD_Lab/Assets/materials/Materials_Samples.usd`

會把原始 Factory paint-line 的材質路徑導向已打包的
`Materials/Materials_Samples.usd`。

2026-07-27 的 Isaac Sim 驗證共解析 135 個 used layer，沒有指向舊 workspace
或其他外部絕對路徑，並找到 2,082 個啟用 collision 的 prim。

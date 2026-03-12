# OSS_DATASET_ID 说明（现已可选）

**更新**：`OSS_DATASET_ID` 现已**可选**。不配置时，代码会使用 **OSS 直接挂载**（`data_source_type="OSS"` + `path=oss://bucket.endpoint/`），无需在 PAI 控制台预创建数据集，与队友流程一致。

若你仍希望使用预创建的数据集，可配置 `OSS_DATASET_ID`，查找方式如下：

## 查找步骤（仅当需要使用预创建数据集时）

1. 打开 **阿里云 PAI-DLC 控制台**：  
   https://pai-dlc.console.aliyun.com/

2. 选择你的 **工作空间**（需与 `DLC_WORKSPACE_ID=464377` 对应）

3. 进入 **「数据管理」** 或 **「数据集」** 菜单

4. 在已有数据集中查看：
   - 数据集 ID 格式为 `d-` 开头的字符串，例如 `d-cvx2t6q7t8w3bnrvgl`
   - 该数据集通常关联 OSS 路径（如 `oss://theta-prod-20260123/`）

5. 若还没有数据集：
   - 点击 **「创建数据集」**
   - 选择类型为 **OSS**
   - 配置 OSS Bucket 与路径
   - 保存后会得到数据集 ID

## 当前默认值

`.env` 中已使用 `d-cvx2t6q7t8w3bnrvgl`。若该 ID 不属于你的工作空间，需在控制台创建或选择正确的数据集后，将得到的 ID 写入 `.env`。

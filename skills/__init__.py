"""AutoLab Agent Lab - Skills 模块

Agent 通过白名单 Skill 与数据/模型交互，禁止直接写 SQL 或修改产模型。

Skill 契约：
- 输入是配置字典或 pandas.DataFrame
- 输出是 pandas.DataFrame / dict / 模型对象，schema 稳定
- 每次调用写 experiment_log
- 只读 Hive、沙箱训练、artifact 落本地目录
"""

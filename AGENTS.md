# 项目约定

- 正式 Python 源码统一放在 `agent/`，入口是 `agent/video_agent.py` 或 `python -m agent`。
- `.env` 保持在项目根目录；不要输出或写入测试产物中的真实 API Key。
- 测试放在 `tests/`。输入故事、历史生成目录和已交付视频不属于源码，不得因重构覆盖。
- 工作流修改必须保留严格审核、候选版本、依赖失效、有界预算和恢复语义；未验收结果不得发布为 `final.mp4`。
- 运行 `conda run --no-capture-output -n vagent python -m unittest discover -s tests -v` 验证。

# 故事转动画视频 Agent

输入 UTF-8 故事文本，程序依次完成故事分析、分镜与帧计划、Canvas 动画绘制、逐帧和时序审核，再通过 Chromium 与 FFmpeg 合成视频。只有最终验收通过才输出 `final.mp4`；中断或审核失败时保留候选版本，可用 `--resume` 继续。

默认视频规格为 **20 秒、24 fps、1280×720、H.264 MP4、无音轨**。当前使用程序化 Canvas 绘制，没有接入图像生成模型、配音或字幕轨道。

## 示例视频

以下四段视频均经过本项目的生成与最终验收流程。README 中显示的是缩小后的 GIF 动态预览；点击预览图或「完整 MP4」可打开原始 **1280×720、24 fps、无音轨** 视频。

| 机械海底世界 · 20 秒 | 小云的彩虹喷嚏 · 6 秒 |
| --- | --- |
| [![机械海底世界动态预览](docs/showcase/sea_world.gif)](docs/showcase/sea_world.mp4) | [![小云的彩虹喷嚏动态预览](docs/showcase/cloud_sneeze.gif)](docs/showcase/cloud_sneeze.mp4) |
| 旧工具、链条和金属零件拼成机械鱼，在幽暗海底游动。 | 小云打了个喷嚏，竟然喷出一道红、黄、蓝三色彩虹。 |
| [完整 MP4](docs/showcase/sea_world.mp4) · [故事原文](story_sea.txt) | [完整 MP4](docs/showcase/cloud_sneeze.mp4) · [故事原文](examples/stories/cloud_sneeze.txt) |

| 气球送星星回家 · 6 秒 | 饼干越狱 · 6 秒 |
| --- | --- |
| [![气球送星星回家动态预览](docs/showcase/balloon_star.gif)](docs/showcase/balloon_star.mp4) | [![饼干越狱动态预览](docs/showcase/cookie_escape.gif)](docs/showcase/cookie_escape.mp4) |
| 红气球牵着小星星升向夜空，小星星停下来眨了眨眼。 | 长出小短腿的巧克力豆饼干，踮脚远离罐子，停下来得意地笑了。 |
| [完整 MP4](docs/showcase/balloon_star.mp4) · [故事原文](examples/stories/balloon_star.txt) | [完整 MP4](docs/showcase/cookie_escape.mp4) · [故事原文](examples/stories/cookie_escape.txt) |

海底视频来自已有的 `sea_video/final.mp4`，展示副本与原成片的 SHA-256 一致。三个新故事保存在 `examples/stories/`；视频规格、来源与验收状态记录在 [`docs/showcase/provenance.json`](docs/showcase/provenance.json)。

<details>
<summary>复现三个新故事的生成命令</summary>

先按下文安装环境并配置 `.env`，再在项目根目录执行：

```bash
conda activate vagent

for story in cloud_sneeze balloon_star cookie_escape; do
  LLM_EXTRA_BODY='{"thinking":{"type":"disabled"},"max_tokens":32768}' \
  python -m agent generate "examples/stories/${story}.txt" \
    --out "outputs/showcase/${story}" --seconds 6 --fps 24 \
    --max-calls 120 --max-seconds 3600
done
```

上述模型参数是本次新示例使用的运行配置，不改写根目录 `.env`；其他服务请按实际支持情况调整。新输出目录须为空，暂停后对同一故事使用原参数追加 `--resume`。本次示例的部分镜头经局部代码修复并提高对应候选上限后重新通过审核，记录见 `provenance.json`。模型输出和修复次数可能随运行变化，复现不保证画面逐像素相同。

</details>

## 1. 环境配置

以下命令在项目根目录执行，使用 Bash / Zsh。已克隆项目时直接进入现有目录，无需再次克隆。

```bash
git clone https://github.com/wflying000/frame_video_agent.git
cd frame_video_agent
```

### 1.1 创建 Python 环境

使用已安装的 Conda 创建 Python 3.10 环境：

```bash
conda create -n vagent python=3.10 -y
conda activate vagent
python -m pip install -r requirements.txt
```

本项目已在 Python 3.10.22 环境验证。Python 依赖为 `playwright==1.51.0` 和 `python-dotenv>=1.0.0`，完整依赖以 [`requirements.txt`](requirements.txt) 为准。

如果已安装 Python 3.10，也可以使用标准虚拟环境，替代上面的 Conda 步骤：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 1.2 安装 Chromium 与 FFmpeg

macOS（需已安装 Homebrew）：

```bash
python -m playwright install chromium --only-shell
brew install ffmpeg
```

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
python -m playwright install --with-deps chromium --only-shell
```

Playwright 的 Chromium 需要单独安装，仅执行 `pip install` 不会安装浏览器。Linux 的 `--with-deps` 会安装浏览器所需的系统依赖，可能需要管理员权限。`ffmpeg` 和 `ffprobe` 都必须在 `PATH` 中。

检查环境：

```bash
python --version
python -m playwright --version
ffmpeg -version
ffprobe -version
python -m agent --help
```

安装完成后，可先执行第 7 节的离线渲染命令，验证浏览器和编码器；该步骤不需要模型密钥。

## 2. 配置模型服务

在项目根目录从模板创建 `.env`；已有 `.env` 时直接编辑，保留现有配置：

```bash
test -f .env || cp .env.example .env
```

模板沿用程序内置的服务地址和模型名称：

```dotenv
LLM_API_KEY=your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_EXTRA_BODY='{"thinking":{"type":"disabled"}}'
```

将 `LLM_API_KEY` 替换为自己的密钥，并按实际服务设置 `LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_EXTRA_BODY`。模板值不代表账号一定具备相应模型权限。生成与审核共用一个模型，该模型必须支持 **Chat Completions、JSON 对象输出和实际多图输入**；仅支持文本的模型无法完成视觉审核。

| 配置项 | 含义 |
| --- | --- |
| `LLM_API_KEY` | API 密钥；未设置时依次读取 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY` |
| `LLM_BASE_URL` | API 基础地址；程序会追加 `/chat/completions`，不要填完整接口路径或 `/anthropic` 协议地址 |
| `LLM_MODEL` | 服务实际支持的多模态模型 ID |
| `LLM_EXTRA_BODY` | JSON 对象，合并进请求体，可覆盖 `max_tokens`、`stream` 等参数；服务专用字段须按实际支持情况填写 |
| `LLM_IDLE_TIMEOUT_SECONDS` | 连续无新增正文或思考文本的超时秒数，默认 `180`，必须是有限正数 |
| `LLM_SHOW_OUTPUT` | 设为 `1` 时实时显示完整模型正文，等价于 CLI 的 `--show-model-output` |

程序默认发送 `temperature=0.2`、`max_tokens=12000`、`response_format={"type":"json_object"}`、`stream=true`。切换服务时，如果不支持模板中的 `thinking` 字段，可将 `LLM_EXTRA_BODY` 改为 `'{}'`；不支持流式时可配置 `'{"stream":false}'`。若输出被截断，应在服务允许范围内调整 `max_tokens` 或思考设置。

`.env` 始终位于项目根目录，不放进 `agent/`。已有进程环境变量优先于 `.env`；纯渲染子进程不读取 `.env`。真实密钥及本地生成产物已在 `.gitignore` 中排除。

终端默认显示阶段进度和模型等待状态。每次调用在候选目录的 `calls/调用名称/累计调用序号/` 保存 `request.json`、`response.txt` 和 `response.json`；请求记录中的图片以哈希表示，响应中已配置的密钥会脱敏。格式错误最多在同一候选内修正一次，仍失败则保留进度并暂停。

## 3. 生成视频的指令

每次新开终端先激活 Python 环境，并进入项目根目录：

```bash
conda activate vagent
# 如果使用 .venv，则执行 source .venv/bin/activate
```

### 3.1 生成简短示例

仓库提供 [`examples/story.txt`](examples/story.txt)：一个红色小球从左向右移动并停下。先用它验证完整生成流程：

```bash
python -m agent generate examples/story.txt \
  --out outputs/red_ball --seconds 4 --fps 24 \
  --show-model-output
```

成功后输出 **`outputs/red_ball/final.mp4`**。生成过程会多次调用模型服务，耗时与费用取决于分镜、审核和修复次数。仅完成部分镜头或审核未通过时不会发布 `final.mp4`。

### 3.2 生成海底故事或自己的故事

使用仓库已有的 [`story_sea.txt`](story_sea.txt)：

```bash
python -m agent generate story_sea.txt \
  --out outputs/sea_video --seconds 20 --fps 24 \
  --show-model-output
```

成功后输出 `outputs/sea_video/final.mp4`。这条命令使用新的 `outputs/sea_video/`，可与本地已有的 `sea_video/` 历史目录并存。

将自己的故事保存为 UTF-8 文本，例如 `my_story.txt`，再执行：

```bash
python -m agent generate my_story.txt \
  --out outputs/my_story --seconds 20 --fps 24
```

也支持文件入口 `python agent/video_agent.py generate ...`。默认输出目录是 `output/`，建议显式指定 `--out outputs/项目名`。首次运行要求输出目录不存在或为空；换故事、改视频时长或帧率时应使用新目录。

### 3.3 断点续跑与预算

恢复时使用相同的故事、输出目录、时长、帧率和模型配置：

```bash
python -m agent generate examples/story.txt \
  --out outputs/red_ball --seconds 4 --fps 24 --resume
```

如果累计预算耗尽，可以明确提高上限后续跑：

```bash
python -m agent generate examples/story.txt \
  --out outputs/red_ball --seconds 4 --fps 24 --resume \
  --max-calls 180 --max-seconds 5400
```

`--resume` 会复用通过且哈希有效的任务，不会清零历史候选次数、模型调用数或活动耗时。没有新版 `manifest.json` 的旧目录不能直接恢复。候选上限和局部修复的处理见第 5 节。

### 3.4 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `generate <故事.txt>` | — | 读取故事并运行模型生成与审核流程 |
| `render <动画.html>` | — | 离线渲染已有 Canvas HTML，无需 API Key |
| `--out` | `output` | 输出目录；新任务必须使用不存在或为空的目录 |
| `--seconds` | `20` | 整数秒，范围 `2～120` |
| `--fps` | `24` | 整数帧率，范围 `1～60` |
| `--resume` | 关闭 | 仅用于 `generate`，恢复已有任务 |
| `--max-calls` | `120` | 整个项目累计模型调用上限，必须大于 0 |
| `--max-seconds` | `3600` | 累计活动耗时上限，必须大于 0，并非严格的墙钟计时截止时间 |
| `--task-attempt-limit TASK=COUNT` | 按任务类型 | 仅用于 `generate --resume`，提高指定任务的累计候选上限，可重复指定 |
| `--show-model-output` | 关闭 | 打印完整模型正文 |
| `--preview` | 关闭 | 仅用于 `render`，只生成采样 PNG，不输出 MP4 |

分辨率固定为 1280×720，当前没有宽高 CLI 参数。视频生成失败或暂停时进程返回非零状态；详细原因见终端、`manifest.json` 和候选目录中的审核记录。

## 4. 生成流程

1. **故事事实与因果**：提取实体、事件、原文依据、前后状态和结局；独立调用审核原文忠实度。
2. **视觉设定与分镜**：定义固定资产与可变状态；检查事件覆盖、顺序、连续整数帧区间以及总帧数。未通过不能进入绘制。
3. **标准参考与文字节奏检查**：逐个绘制并审核角色/道具/场景参考。用完整分镜文字核对镜头顺序、时长、动作节奏和状态衔接，保存 `plans/pacing.txt`；此阶段不再生成文字占位 PNG、HTML 或 `animatic.mp4`。实际动作仍由后续画面审核验证。
4. **每镜头帧计划**：规划首尾、动作接触、状态切换、关键帧与动作区间。关键帧必须覆盖镜头硬性要求，时间线采用从 0 开始的整数帧和半开区间。
5. **独立镜头绘制与逐帧验收**：生成该镜头的确定性 Canvas 代码；先渲染并审核锚定帧，再逐个处理其余关键帧。审核传入实际标准图、当前图和必要的前帧。
6. **局部纠错与时序审核**：失败优先修改当前镜头的代码片段，再整镜头复验。采样覆盖全部关键帧、接触/切换前后邻域、镜头边界及普通区间；逐项检查动作、因果、接触与身份连续性。
7. **独立镜头导出与边界审核**：重新加载时核对预览一致性，逐帧生成 PNG，编码各镜头；再独立检查相邻镜头的状态衔接。
8. **合成与最终验收**：按时间线拼接，检查实际解码的尺寸、帧率、总帧数、时间戳和未声明黑屏。从实际成片取图，先不提供故事进行独立观察，再逐事件对照原文验收并记录证据帧。全部必需任务通过且版本有效才发布 `final.mp4`。

审核状态使用 `PASS / FAIL / UNCERTAIN / NOT_APPLICABLE`。缺项、版本不匹配、硬要求的 `UNCERTAIN` 或 `NOT_APPLICABLE` 都不能放行；视觉 PASS 必须引用本次实际提供的帧号。

## 5. 局部修复、版本与恢复

每个任务的候选都保存在独立的 `v001/`、`v002/` 目录。审核绑定目标 ID 和版本；清单保存输入哈希、依赖版本、输出哈希、候选次数和模型调用数。

镜头默认最多生成 5 个候选。失败后基于最近实际保存的 `scene.html` 持续局部修复，并沿用该版本有效的帧计划；不会仅因到第 4、5 轮而整镜头重画。只有审核明确指出计划错误时才重规划。补丁失败时回退到最近有实际代码的候选，避免把上一轮补丁 JSON 当作完整代码。规划与资产任务默认最多 3 个候选；全片验收最多触发两次定位后的镜头修复。重试和全片修复不通过重启或换候选版本清零。网络错误有限重试一次，随后保留候选和已完成调用供恢复。

流式调用默认连续 180 秒没有新增正文或思考文本即超时，SSE 保活、空增量和用量事件不会重置该计时；可用 `LLM_IDLE_TIMEOUT_SECONDS` 设置有限正数。收到完成原因后直接结束读取，避免继续等待连接关闭。超时保留部分正文与诊断，同一候选内最多重试一次，连续超时则暂停供 `--resume` 恢复，不消耗新的画面候选。

服务返回 HTTP 429 或 5xx 时，至少等待 5 秒后有限重试一次；若 `Retry-After` 指定更长秒数，则遵循该值，超过 30 秒时保存进度并提示稍后恢复。连续服务错误会明确暂停，已通过的审核和已生成代码可复用，恢复不会重新占用一个候选。

局部修复同时传入原始全图和失败区域的真实裁切放大图，附带原始帧号、区域坐标与哈希。新镜头可基于上一已通过镜头的代码和末帧实现后续动作，减少造型和背景跳变。每次修复后仍重新审核全部关键帧、时序和邻接关系；不会用局部放大图代替整帧验收。候选目录保存 `repair_strategy.json`、`repair_evidence/` 或 `continuity_reference.json` 以追溯修复依据。

单份失败报告的修复输入最多选取 6 张实际帧，每帧最多放大两个失败区域；原始完整审核记录仍保留，避免重复发送大量图片。单镜头时序审核检查当前末态是否符合下一镜头已审核的进入计划，不索取尚未生成的下一镜头图片。所有镜头完成后，仍必须通过基于两侧真实帧的独立边界审核。

边界审核结合两侧分镜的机位和景别，区分计划允许的视角变化与剧情状态跳变；切镜不强制屏幕坐标、尺寸和正背面完全相同。角色身份、物体开闭和动作结果仍须连续。

边界审核从两段实际编码视频各取紧贴切点的三张连续帧，并记录切点与邻帧的像素变化量。新增动作相位检查，不能把机位描述变化当作身体形变、头部伸出、尾部相位归零或背景突然消失的通用理由；变化量只是定位线索，仍需按真实画面审核。边界修复会获得上一镜头的实际 HTML、最后帧时间和下一采样时间，以连续动作参数承接局部 `t=0`，禁止用两幅身体交叉淡化掩盖跳变。旧版边界审核需复验；镜头历史和旧成片保留。

边界审核同时接收两侧已审核的帧计划与实际采样的局部帧号，只判断切点邻域当时应有的动作状态；不会要求计划稍后发生的动作提前在首三帧完成。当前时刻应有的道具、连接与姿态仍需通过真实画面检查，完整动作继续由镜头时序和成片事件审核验证。

明确定位到代码问题时，修复请求也可携带 `local_patch`（项目内补丁路径、补丁哈希与源 HTML 哈希），应用人工编写的精确片段替换。源版本或补丁内容改变就拒绝应用；新候选记录 `applied_local_patch.json`，不伪装成模型输出，不消耗生成调用，但仍须完成全部逐帧、时序、边界和成片审核。候选次数和审核调用照常计入预算。

若自动细化的首帧要求与连续衔接冲突，可针对明确的候选版本提交 `scope: plan` 与 `replan_source_version`。重规划参考真实失败帧和上一镜头末帧，仍独立审查原分镜全部硬要求；之后先用新计划完整审核现有代码，确实不满足时才进入代码修复。该请求只作用于指定源版本，不在后续失败候选中无限重新规划。

首帧要求显式比较上一镜头时，像素审核同时提供上一镜头实际末帧及其哈希，并与当前帧、标准资产分开标注。旧失败审核缺少这项证据时，可在下一候选补图复核一次；已经提供过参考图仍失败的情况不会无限复核。过期本地补丁会立即暂停，避免自动重试耗尽后续候选。

如果当前失败的关键帧曾经通过，会提供最近通过版本到当前实际 HTML 的代码差异（最多 24,000 字符）；长单行 Canvas 代码会提取精确变化片段，避免整行差异过长而丢失依据。同时附带最多一张与原审核请求哈希一致的历史通过帧，帮助定位被后续补丁误删的变换或图层。远景、中景和近景分别控制可见性，不能为隐藏近景而让全部背景消失。历史通过图片只作修复参考，仍需保留其他已完成的修复并对新候选重新验收。

关键帧审核根据要求中给定的坐标附带真实局部图，时序与最终事件审核最多补充四个关键帧细节。放大图不会再被当作全尺寸原图裁剪。若要求、上下文、实际图片和模型配置完全一致，可以复用旧的视觉 PASS，并在 `reused_from` 中记录原报告路径、哈希和版本；图片或要求变化则重新审核，FAIL/UNCERTAIN 不会被复用为通过。修复请求携带最近六轮反馈，避免在不同错误之间来回撤销修复。

关键帧审核明确区分当前帧、比较帧和造型参考图。若首次失败明确给出了实际帧号和局部区域，额外提供最多四张真实区域放大图，按全部原始要求复核一次。两次报告、区域图片及其哈希均保留在候选目录；第二次仍失败或不确定就进入修复，不会反复请求直到通过。

对于“相对前帧向右下摆”等方向要求，审核必须区分跨帧位移与最终位置。局部部位用同一主体锚点比较两帧相对坐标，避免把主体平移当摆动，也不能因尾端仍在躯干左侧就否定其向右移动。明确要求“位于右下方”的静态约束仍按终态方位检查。旧版方向失败若缺少该参照规则，下一候选先保留实际代码、按新规则复验一次；复验仍失败才依据新报告继续修复。历史报告不改写，所有硬要求、像素证据与候选/调用预算仍生效。

“由右下摆改为左后上摆”等句式中的起始、结束相位采用同一方向规则；不能把起始相位偷偷换成前帧的绝对方位。一张静态前帧只能显示位置，不能独自证明或否定此前的运动方向。

时序审核中的坐标测量使用一致参考点：原文明示部位就测量该部位，仅指定角色中心时使用主要躯干中心，不把摆动尾鳍、背鳍或伸出工具的整体包围框中心当作路径锚点。起点、过渡和终点都须根据实际像素观测，无法定位仍返回 `UNCERTAIN`。

最终独立观察逐项校验 `frame_index`、非空描述及完整帧号顺序。字段拼写、类型或覆盖错误会保存原响应和结构诊断，在同一候选内附带相同真实图片有限修正一次；不会自动猜测或改写帧号。连续格式失败暂停并保留候选，结构无效的缓存标记为 `SCHEMA_ERROR`，恢复时不会复用为有效结果。

时序审核除关键帧及其邻域外，还采样每个动作区间内部的 1/4、1/2、3/4 时间点，以及保持区间中点，避免只在锚点正确而过渡中出现重影。姿态修复要求使用同一主体的形状和位置变化，不能把两幅完整身体交叉淡化当作动作。

局部修复以**镜头代码**为最小重新生成单位。标准资产保持独立；通过的其他镜头直接复用。修改某个镜头后，相关边界审核、合成和全片验收失效并重跑。已发布视频失效时会移至 `exports/previous_final_*.mp4`，避免把旧成片当作新版本交付。

中断、网络失败或预算不足后，用相同故事、时长和帧率恢复：

```bash
python -m agent generate my_story.txt \
  --out outputs/my_story --seconds 20 --fps 24 --resume
```

首次运行要求空目录。恢复会检查实际文件哈希；已通过且输入未变的任务不调用模型。已生成代码但审核中断时，会继续该候选的后续步骤。修改故事或视频规格请使用新目录。

如果提示 `assets/L01 已达到 3 个候选版本上限`，单独加 `--resume` 不会清零次数。修复失败原因后，可以显式为这个任务增加一个候选：

```bash
python -m agent generate my_story.txt --out outputs/my_story --seconds 20 --fps 24 \
  --resume --task-attempt-limit assets/L01=4
```

`4` 是累计总上限，下一个候选写入 `v004/`，历史失败报告保留。指定上限会保存到清单，后续恢复继续沿用；其他任务仍使用原来的上限，模型调用数和活动耗时也不会清零。多个任务可重复传此参数。

恢复时应保持原 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_EXTRA_BODY` 配置，否则输入签名变化会要求重新生成已通过任务。新项目可以按服务支持范围另设 `max_tokens`。

对于 `frame(0.00) 重画不一致`，渲染器现已在生成代码执行前为 Canvas 2D 设置 `willReadFrequently: true`，避免 Chromium 在多次 PNG 读回时切换绘制后端造成像素变化。仍严格检查逆序重画、重新加载和预览一致性；真正使用随机数、时钟或累计状态的动画仍会拒绝，增加 `max_tokens` 不能修复这种绘制问题。

如果帧计划的首帧仅描述进入状态，模型可能返回 `covers: []` 和 `requirements: []`。程序会从该帧已有的 `state` 补出一项 `hard: true` 的静态画面检查，保留原始 `generated.json`，在 `normalized.json` 和 `normalization.json` 记录规范化结果与来源。该状态帧仍需逐帧视觉审核；不会虚构已完成的动作，也不会放宽镜头硬性要求覆盖。缺失字段、错误类型或已有动作覆盖却没有检查项，仍会报告具体字段路径。该处理也适用于镜头局部重规划。

若 `plans/frames/S03` 已因空检查项耗尽三个候选，修复后在原模型配置下使用 `--resume --task-attempt-limit plans/frames/S03=4` 继续；已保存的其他任务上限会沿用，无需再次传入 `assets/L01=4`。

项目默认累计上限为 120 次模型调用、3600 秒活动耗时。时间预算在调度和模型调用检查点检查，单次在途网络/渲染请求另有超时；它不是实时强制终止器。需要增加预算时显式指定：

```bash
python -m agent generate my_story.txt --out outputs/my_story --resume \
  --seconds 20 --fps 24 --max-calls 180 --max-seconds 5400
```

只有验收完成才标记 `COMPLETED`；仍有失败、不确定或预算不足时标记 `BLOCKED`，保留材料并返回错误。输出目录有单写者锁，不能并发运行同一个项目。

## 6. 代码与产物

| 正式代码 | 职责 |
| --- | --- |
| `agent/video_agent.py`、`agent/__main__.py` | 文件入口、模块入口和 CLI 参数 |
| `agent/workflow.py` | 文字节奏检查、阶段调度、镜头衔接和成片验收 |
| `agent/planning.py`、`agent/prompts.py` | 故事/视觉/分镜/帧规划、独立角色提示词 |
| `agent/drawing.py` | 标准资产、逐关键帧验收和镜头局部修复 |
| `agent/repair.py` | 选择实际修复代码、沿用有效帧计划、准备真实局部证据 |
| `agent/validation.py` | 结构、覆盖、帧时间线和审核硬门槛 |
| `agent/store.py` | 任务版本、依赖哈希、预算、锁和断点恢复 |
| `agent/model.py` | 根目录配置、流式模型请求和实际图片输入 |
| `agent/rendering.py` | 确定性绘帧、镜头编码、拼接与解码检查 |
| `agent/common.py` | 日志、原子 JSON 写入和内容哈希 |

| 输出路径 | 内容 |
| --- | --- |
| `project_config.json`、`story.txt` | 制作参数和不可覆盖的原始故事 |
| `manifest.json` | 项目状态、预算、版本、依赖与审核绑定 |
| `plans/` | 故事事实、因果、视觉设定、分镜、事件覆盖和帧计划 |
| `assets/<资产ID>/vNNN/` | 标准参考代码、实际参考图和审核 |
| `plans/pacing.txt` | 完整文字时间线：镜头时长、帧区间、动作、前后状态与检查项 |
| `plans/animatic/vNNN/` | 文字节奏检查候选与审核；沿用历史任务 ID 以兼容恢复 |
| `segments/<镜头ID>/vNNN/` | 镜头代码、帧任务、关键帧审核、时序审核、PNG 和视频 |
| `checks/timeline/vNNN/` | 镜头边界审核 |
| `exports/final/vNNN/` | 候选成片、真实成片采样、独立观察、事件证据和验收报告 |
| `final.mp4` | 当前有效、已验收的最终视频 |

`plans/` 下的汇总 JSON 便于阅读；恢复以版本目录和 `manifest.json` 为准。不要通过编辑汇总文件来修改计划。

已有项目恢复时，仍复用输入、依赖及产物校验均有效的 `plans/animatic` 审核，并从同一份分镜导出 `plans/pacing.txt`，不额外调用模型或重新渲染占位图。历史 `animatic.mp4` 和版本目录中的图片会保留，但新流程不再生成或读取这些占位画面来审核新候选；候选次数、预算和后续视频视觉审核保持原有规则。

## 7. 离线渲染与验证

已有 HTML 可以单独渲染，无需 API Key。仓库提供 [`examples/rope.html`](examples/rope.html)，以下命令可验证 Chromium 与 FFmpeg 是否正常工作：

```bash
# 渲染完整 20 秒示例，输出 outputs/rope_render/video.mp4
python -m agent render examples/rope.html \
  --out outputs/rope_render --seconds 20 --fps 24

# 只生成采样预览，不编码视频
python -m agent render examples/rope.html \
  --out outputs/rope_preview --seconds 20 --fps 24 --preview
```

离线渲染输出的是 `video.mp4`，不经过故事生成流程的模型审核；`generate` 的验收成片才叫 `final.mp4`。`render` 不支持 `--resume`，重复运行时请换新的空输出目录。

执行项目测试：

```bash
conda run --no-capture-output -n vagent python -m unittest discover -s tests -v
# 或在已激活的 Python 环境中执行：
python -m unittest discover -s tests -v
```

测试覆盖流式响应、超时与服务重试、审核门槛、事件与时间线校验、缓存失效、预算与恢复、局部修复、最终验收回退，以及实际 Chromium 绘帧、FFmpeg 编码与解码检查。模型响应由测试替身提供，不会调用真实模型 API；测试通过不能保证真实模型对所有故事都能生成合格成片。

## 8. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `ModuleNotFoundError` | 确认已激活安装依赖时使用的环境，再执行 `python -m pip install -r requirements.txt` |
| Chromium executable 不存在或缺系统库 | 在同一环境重新执行浏览器安装命令；Linux 使用 `--with-deps` |
| 提示安装 FFmpeg / ffprobe | 安装 FFmpeg，并确认当前终端能运行 `ffmpeg -version` 和 `ffprobe -version` |
| API 401 / 403、模型不存在、图片输入不支持 | 核对 `.env` 中的密钥、基础地址、模型权限与多图输入能力；确认没有旧的环境变量覆盖 `.env` |
| 输出被截断或 JSON 解析失败 | 查看候选的 `calls/` 诊断；按服务限制调整输出额度或思考配置；更改模型配置后可能需要重新生成已通过任务 |
| HTTP 429 / 5xx 或连续超时 | 程序有限重试后保留进度，待服务恢复后按原参数追加 `--resume` |
| 输出目录非空 | 恢复生成任务时使用 `--resume`；新任务或离线渲染使用新的空目录 |
| 候选次数或调用预算耗尽 | 先检查失败原因，再用第 3、5 节的预算或候选上限参数续跑 |
| 没有 `final.mp4` | 检查 `manifest.json` 是否为 `COMPLETED`；审核未通过或预算不足时状态为 `BLOCKED`，不会发布未验收成片 |

建议将新产物统一放在 `outputs/`。`.gitignore` 排除了 `.env`、本地环境、缓存、`output/`、`outputs/` 和现有 `sea_video/`；生成过程的历史版本及诊断留在本机，仓库保存源码、测试、故事和配置模板；已验收的展示视频与 GIF 单独保存在 `docs/showcase/`。使用其他输出目录时，请自行添加对应的忽略规则。

当前局部修复以镜头代码为单位，可能影响整个镜头，因此每次修复都需重新验收。视觉审核使用带帧号的有序图片，仍可能漏掉短暂细节，交付前应观看完整视频。

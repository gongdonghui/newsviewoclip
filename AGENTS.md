# AGENTS.md

## 1. 项目目标

本项目用于将一段较长的视频自动或半自动拆分为多个独立短视频片段。

Agent 的核心职责：

1. 读取本地长视频文件。
2. 提取音频并生成转录文本。
3. 根据语义、停顿、主题变化、关键词或用户给定规则，识别适合剪出的片段。
4. 输出剪辑建议清单，供用户确认。
5. 使用 FFmpeg 执行无损或高质量剪辑。
6. 导出每个片段的视频文件、字幕文件和元数据清单。
7. 保留原始视频，不覆盖源文件。

适用场景包括：

- 新闻播报拆条
- 课程视频切分
- 访谈精华片段提取
- 播客视频切片
- 短视频二次剪辑
- 产品演示视频拆分
- 会议录像按议题切分

---

## 2. 默认工作原则

### 2.1 安全原则

- 永远不要覆盖原始视频。
- 所有输出写入独立目录。
- 执行 FFmpeg 命令前，先生成剪辑方案并展示。
- 如果片段边界不明确，优先保留更多上下文。
- 不自动删除临时文件，除非用户明确要求。
- 不对视频内容做事实判断，只做内容切分和结构化整理。
- 如果处理失败，保留日志并给出可复现的命令。

### 2.2 默认目录结构

```text
project/
├── AGENTS.md
├── input/
│   └── source.mp4
├── output/
│   ├── clips/
│   ├── subtitles/
│   ├── thumbnails/
│   ├── metadata/
│   └── logs/
├── temp/
└── scripts/
```

默认源视频路径：

```text
input/source.mp4
```

默认输出目录：

```text
output/
```

---

## 3. 推荐技术栈

优先使用以下本地工具：

- Python 3.11+
- FFmpeg
- ffprobe
- Whisper 或 faster-whisper
- 可选：PySceneDetect
- 可选：MoviePy，仅用于简单封装，不作为首选剪辑引擎
- 可选：OpenCV，用于生成缩略图或镜头检测

优先级：

1. FFmpeg：视频剪辑、音频提取、字幕烧录、格式转换
2. ffprobe：读取视频元数据
3. faster-whisper：本地语音转录
4. Python：生成剪辑计划、清单和批处理脚本
5. PySceneDetect：检测画面切换，辅助优化片段边界

---

## 4. Agent 工作流程

## 阶段 A：检查运行环境

先检查以下命令是否可用：

```bash
ffmpeg -version
ffprobe -version
python --version
```

如果需要转录，再检查：

```bash
python -c "import faster_whisper; print('faster-whisper installed')"
```

如果未安装依赖，给出安装建议，但不要擅自修改系统环境。

推荐安装方式：

```bash
pip install faster-whisper
```

可选依赖：

```bash
pip install scenedetect opencv-python
```

---

## 阶段 B：读取视频信息

使用 ffprobe 读取：

- 文件名
- 视频总时长
- 分辨率
- 帧率
- 视频编码
- 音频编码
- 音频采样率
- 文件大小

示例：

```bash
ffprobe -v quiet \
  -print_format json \
  -show_format \
  -show_streams \
  input/source.mp4
```

将结果保存为：

```text
output/metadata/source_probe.json
```

---

## 阶段 C：提取音频

从长视频提取单声道 16kHz WAV 音频，供 Whisper 转录：

```bash
ffmpeg -i input/source.mp4 \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  temp/source_audio.wav
```

如果文件已存在，先检查是否可复用。

---

## 阶段 D：生成转录文本

使用本地 Whisper 或 faster-whisper 输出带时间戳的转录结果。

建议至少生成：

```text
output/metadata/transcript.json
output/subtitles/source.srt
output/subtitles/source.vtt
output/metadata/transcript.txt
```

转录 JSON 中，每段至少包含：

```json
{
  "id": 1,
  "start": 12.45,
  "end": 18.92,
  "text": "这里是转录文本"
}
```

如果用户未指定语言，优先自动识别。

如果用户指定中文、英文或其他语言，应明确设置语言参数。

对于新闻、课程、访谈类视频，优先保留标点并进行轻度文本整理，但不要改写原意。

---

## 阶段 E：识别候选片段

根据视频类型选择切分策略。

### 5.1 新闻播报

切分依据：

- 每条新闻主题变化
- 主播明显转场表达
- 新闻标题或字幕变化
- 人名、地点、事件主体变化
- 画面切换
- 片段通常控制在 20 秒至 180 秒

常见转场表达：

```text
接下来关注
来看下一条消息
此外
与此同时
再来看
下面来看
最后关注
```

### 5.2 访谈或播客

切分依据：

- 一个完整观点
- 一个问题与对应回答
- 明显情绪高潮
- 适合传播的金句
- 话题切换
- 片段通常控制在 30 秒至 180 秒

### 5.3 课程或讲解视频

切分依据：

- 一个知识点
- 一个例题
- 一个概念定义
- 一个操作步骤
- 一个章节
- 片段通常控制在 60 秒至 600 秒

### 5.4 产品演示

切分依据：

- 一个功能点
- 一个页面流程
- 一个操作步骤
- 一个业务场景
- 片段通常控制在 20 秒至 180 秒

---

## 6. 剪辑边界规则

片段边界需要兼顾语义完整性与观看体验。

默认规则：

- 起点向前预留 0.3 至 1.0 秒。
- 终点向后预留 0.5 至 1.5 秒。
- 避免从一句话中间开始。
- 避免在一句话尚未说完时结束。
- 尽量在自然停顿、画面转场或完整句末尾切分。
- 如果检测到镜头切换，可在语义边界附近优先对齐镜头边界。
- 两个相邻片段可以保留少量重叠，默认不超过 1 秒。
- 片段过短时，优先与前后片段合并。
- 片段过长时，按子主题再次拆分。

默认时长建议：

```text
最短片段：15 秒
推荐片段：30 至 180 秒
最长片段：300 秒
```

用户指定时长要求时，以用户要求为准。

---

## 7. 先生成剪辑计划，不直接剪辑

在正式执行剪辑前，必须先生成：

```text
output/metadata/clip_plan.json
output/metadata/clip_plan.csv
output/metadata/clip_plan.md
```

每个片段至少包含：

```json
{
  "clip_id": "clip_001",
  "title": "片段标题",
  "start": "00:01:12.300",
  "end": "00:02:05.800",
  "duration_seconds": 53.5,
  "summary": "本片段主要内容",
  "keywords": ["关键词1", "关键词2"],
  "reason": "为什么建议剪出该片段",
  "confidence": 0.92,
  "output_file": "output/clips/clip_001.mp4"
}
```

`clip_plan.md` 使用便于人工确认的表格：

```markdown
| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |
|---|---|---:|---:|---:|---|---|
| clip_001 | 示例标题 | 00:01:12.300 | 00:02:05.800 | 53.5 秒 | 示例摘要 | 主题完整，边界清晰 |
```

如果用户没有明确要求自动执行，默认只生成剪辑计划，不直接批量剪辑。

---

## 8. 执行剪辑

用户确认剪辑计划后，生成并执行 FFmpeg 命令。

### 8.1 快速无重编码剪辑

适合快速输出，但切点可能会受关键帧影响：

```bash
ffmpeg -ss 00:01:12.300 \
  -to 00:02:05.800 \
  -i input/source.mp4 \
  -c copy \
  output/clips/clip_001.mp4
```

### 8.2 精确剪辑

适合短视频发布，切点更加准确：

```bash
ffmpeg -ss 00:01:12.300 \
  -to 00:02:05.800 \
  -i input/source.mp4 \
  -c:v libx264 \
  -preset medium \
  -crf 20 \
  -c:a aac \
  -b:a 192k \
  -movflags +faststart \
  output/clips/clip_001.mp4
```

默认优先使用精确剪辑模式。

如果需要批量处理，应先生成 shell 脚本：

```text
scripts/cut_clips.sh
```

并将实际执行日志写入：

```text
output/logs/cut_clips.log
```

---

## 9. 导出字幕

每个视频片段应尽量导出对应字幕：

```text
output/subtitles/clip_001.srt
output/subtitles/clip_001.vtt
```

字幕时间戳需要从全局时间轴转换为片段内相对时间轴。

例如，原字幕时间：

```text
00:01:15.000 --> 00:01:19.000
```

如果片段起点为：

```text
00:01:12.000
```

则片段字幕应转换为：

```text
00:00:03.000 --> 00:00:07.000
```

如用户要求烧录字幕，可使用：

```bash
ffmpeg -i output/clips/clip_001.mp4 \
  -vf "subtitles=output/subtitles/clip_001.srt" \
  -c:v libx264 \
  -crf 20 \
  -preset medium \
  -c:a copy \
  output/clips/clip_001_burned.mp4
```

默认不烧录字幕，只输出独立字幕文件。

---

## 10. 生成缩略图

每个片段默认截取一张缩略图。

优先选择：

- 片段开始后 2 至 5 秒
- 画面清晰
- 无黑屏
- 无快速转场
- 人物表情自然
- 标题画面完整

示例：

```bash
ffmpeg -ss 00:00:03 \
  -i output/clips/clip_001.mp4 \
  -frames:v 1 \
  output/thumbnails/clip_001.jpg
```

---

## 11. 用户可配置参数

建议在项目根目录创建：

```text
config.json
```

示例：

```json
{
  "input_video": "input/source.mp4",
  "output_dir": "output",
  "mode": "news",
  "language": "auto",
  "min_clip_seconds": 20,
  "target_clip_seconds": 90,
  "max_clip_seconds": 180,
  "padding_before_seconds": 0.5,
  "padding_after_seconds": 1.0,
  "generate_subtitles": true,
  "generate_thumbnails": true,
  "burn_subtitles": false,
  "cut_mode": "accurate",
  "auto_execute": false
}
```

支持的 `mode`：

```text
news
interview
podcast
course
product_demo
meeting
custom
```

支持的 `cut_mode`：

```text
fast
accurate
```

---

## 12. 推荐脚本

建议拆分为以下脚本：

```text
scripts/
├── probe_video.py
├── extract_audio.py
├── transcribe.py
├── detect_scenes.py
├── build_clip_plan.py
├── cut_clips.py
├── export_clip_subtitles.py
├── generate_thumbnails.py
└── run_pipeline.py
```

### 12.1 run_pipeline.py

统一入口建议：

```bash
python scripts/run_pipeline.py \
  --input input/source.mp4 \
  --mode news \
  --language auto \
  --plan-only
```

确认剪辑计划后：

```bash
python scripts/run_pipeline.py \
  --input input/source.mp4 \
  --mode news \
  --execute
```

---

## 13. 代码风格要求

- Python 版本：3.11+
- 使用 `pathlib.Path`
- 使用类型标注
- 使用 `argparse`
- 使用 `subprocess.run(..., check=True)`
- 日志统一使用 `logging`
- 时间转换函数单独封装
- 对输入文件、输出目录和外部命令做检查
- 对异常情况输出清晰错误信息
- 不要将所有逻辑写在单个文件中
- 不要引入不必要的重量级依赖
- 脚本应支持断点续跑
- 已存在文件默认跳过，除非传入 `--force`

---

## 14. Agent 执行任务时的响应格式

每次处理视频时，按以下顺序反馈：

### 第一步：视频概况

```text
视频路径：
视频时长：
分辨率：
音轨情况：
预计输出目录：
```

### 第二步：切分策略

```text
视频类型：
切分依据：
建议片段时长：
是否需要字幕：
是否仅生成剪辑计划：
```

### 第三步：候选片段

展示 `clip_plan.md` 中的候选片段表格。

### 第四步：执行结果

```text
已生成片段数量：
成功数量：
失败数量：
输出目录：
字幕目录：
缩略图目录：
日志路径：
```

---

## 15. Agent 接收用户指令的示例

### 示例 1：新闻视频拆条

```text
请分析 input/news.mp4。
按照每条新闻拆分。
每段控制在 30 秒到 2 分钟。
先生成剪辑计划，不要直接剪。
```

### 示例 2：访谈提取短视频

```text
请分析 input/interview.mp4。
提取适合发短视频的观点片段。
每段 45 秒到 90 秒。
优先保留完整观点和金句。
生成字幕和缩略图。
```

### 示例 3：课程按知识点切分

```text
请分析 input/course.mp4。
按照知识点切分。
每段尽量控制在 3 到 8 分钟。
需要输出标题、摘要、字幕和剪辑文件。
```

### 示例 4：按时间点直接剪辑

```text
请从 input/source.mp4 中剪出以下片段：
1. 00:01:10 到 00:02:25
2. 00:05:40 到 00:06:30
3. 00:09:15 到 00:10:05

使用精确剪辑模式。
输出到 output/clips。
```

---

## 16. Agent 的优先决策规则

遇到多个可选方案时，按以下顺序决策：

1. 保证原始视频安全。
2. 保证剪辑边界完整。
3. 优先生成可人工确认的剪辑计划。
4. 优先使用本地工具。
5. 优先使用 FFmpeg。
6. 优先输出结构化元数据。
7. 优先支持断点续跑。
8. 优先给出可复现命令。
9. 遇到不明确条件时，使用默认值继续执行，并在结果中说明。
10. 不因某个可选依赖缺失而中断整个流程。

---

## 17. 最小可交付版本

首次实现时，至少完成以下能力：

1. 读取视频元数据。
2. 提取音频。
3. 使用 faster-whisper 转录。
4. 输出带时间戳的 transcript.json。
5. 根据转录结果生成 clip_plan.json、clip_plan.csv 和 clip_plan.md。
6. 用户确认后使用 FFmpeg 输出多个视频片段。
7. 为每个片段输出独立字幕。
8. 记录处理日志。

后续增强功能：

- 镜头切换检测
- 自动标题生成
- 自动关键词提取
- 自动缩略图优选
- 竖屏裁切
- 字幕烧录
- 人脸跟踪
- 静音检测
- 热点片段评分
- GUI 或 Web 页面
- 批量处理多个视频

---

## 18. 当前默认任务

当用户只说“帮我从长视频中剪出一段一段的视频”时，默认执行：

```text
输入文件：input/source.mp4
模式：custom
策略：先转录，再按语义完整性切分
最短片段：20 秒
目标片段：60 至 120 秒
最长片段：180 秒
切点缓冲：前 0.5 秒，后 1.0 秒
字幕：生成独立 SRT 和 VTT
缩略图：生成
剪辑模式：accurate
是否自动执行：否，先输出剪辑计划
```

在用户确认剪辑计划后，再执行批量剪辑。

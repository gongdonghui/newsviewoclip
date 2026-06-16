# news_video_clip_agent

一个面向新闻长视频拆条的本地剪辑项目。

项目目标不是做通用 NLE，而是把一条 20 到 30 分钟左右的新闻节目，自动拆成一组更短、更集中、可单独分发的视频片段，并同时产出字幕和结构化元数据。

## 主要功能

- 读取本地视频并提取元数据
- 提取单声道 16kHz WAV 音频
- 使用本地 `mlx_whisper` 生成带时间戳转录
- 优先使用 `DeepSeek` 做新闻语义拆条
- 在语义边界附近结合语音停顿修正 clip 结尾
- 回退到本地启发式分组，保证没有 LLM 也能出方案
- 导出 `clip_plan.json`、`clip_plan.csv`、`clip_plan.md`
- 执行 FFmpeg 精确剪辑
- 为每个 clip 导出独立 `srt` / `vtt`
- 为每个输入视频使用独立输出目录，避免不同视频互相覆盖
- 对同一输入加运行锁，避免并发跑乱产物

## 当前实现的剪辑策略

当前项目最适合 `news` 模式，核心策略是：

1. 先转录，不直接按镜头切  
   入口在 [scripts/transcribe.py](/Users/gongshuai/workspace/english-video-clip/scripts/transcribe.py:1)。当前实现使用本地 `mlx_whisper`，把整条视频转成带时间戳的 transcript。

2. 优先用 DeepSeek 做语义选段  
   入口在 [scripts/analyze_clips.py](/Users/gongshuai/workspace/english-video-clip/scripts/analyze_clips.py:1)。  
   当前 prompt 重点是：
   - 英文新闻
   - 目标片长 `15-45s`
   - 优先主播主导片段
   - 允许很短的记者/采访延续
   - 偏好更紧、更短的 clips，而不是长 package

3. DeepSeek 失败时回退到本地启发式分组  
   逻辑在 [scripts/build_clip_plan.py](/Users/gongshuai/workspace/english-video-clip/scripts/build_clip_plan.py:1)。  
   fallback 不是按镜头切，而是按 transcript 段、时长阈值和间隔做分组。

4. 用停顿检测修正边界  
   逻辑在 [scripts/detect_pauses.py](/Users/gongshuai/workspace/english-video-clip/scripts/detect_pauses.py:1)。  
   项目会调用 `ffmpeg silencedetect` 识别语音停顿，并把 clip end 吸附到更自然的停顿附近，尽量避免在主播一句话中间收尾。

5. 最后做统一 padding 和导出  
   在 [scripts/run_pipeline.py](/Users/gongshuai/workspace/english-video-clip/scripts/run_pipeline.py:1) 里统一做：
   - 前补 `0.5s`
   - 后补 `1.0s`
   - 输出 clip、字幕和 plan

## 适合的内容类型

- 英文电视新闻
- 主播串联 + 记者 package 的晚间新闻
- 需要快速拆成多个短视频的资讯类节目

不太适合：

- 纯访谈长对话
- 课程录播
- 强依赖镜头语言的纪录片
- 需要按镜头、人物或情绪做精细剪辑的内容

## 技术栈

- Python 3.11+
- FFmpeg / ffprobe
- `mlx_whisper`
- DeepSeek Chat Completions API

## 项目结构

```text
news_video_clip_agent/
├── README.md
├── AGENTS.md
├── docs/
├── scripts/
│   ├── analyze_clips.py
│   ├── build_clip_plan.py
│   ├── common.py
│   ├── cut_clips.py
│   ├── detect_pauses.py
│   ├── export_clip_subtitles.py
│   ├── extract_audio.py
│   ├── probe_video.py
│   ├── run_pipeline.py
│   └── transcribe.py
├── tests/
├── temp/
└── output/
```

说明：

- `temp/` 是中间产物目录，默认不提交
- `output/<run_id>/` 是每个输入视频对应的独立输出目录

## 依赖准备

先确认本地环境：

```bash
ffmpeg -version
ffprobe -version
python3 --version
```

如果需要 DeepSeek 语义拆条，准备环境变量：

```bash
export DEEPSEEK_API_KEY=your_key
```

如果没有这个变量，项目仍然可以运行，但会直接走本地 fallback 分组。

## 常用命令

只生成剪辑方案：

```bash
python3 scripts/run_pipeline.py \
  --input ~/Downloads/l13.mp4 \
  --mode news \
  --language en \
  --plan-only
```

生成方案并实际剪辑：

```bash
python3 scripts/run_pipeline.py \
  --input ~/Downloads/l13.mp4 \
  --mode news \
  --language en \
  --execute
```

强制重跑中间产物：

```bash
python3 scripts/run_pipeline.py \
  --input ~/Downloads/l13.mp4 \
  --mode news \
  --language en \
  --plan-only \
  --force
```

## 输出内容

每次运行会生成一个独立目录：

```text
output/<run_id>/
├── clips/
├── subtitles/
├── metadata/
└── logs/
```

其中：

- `metadata/source_probe.json`
  - 视频元数据
- `metadata/transcript.json`
  - 带时间戳转录
- `metadata/pauses.json`
  - 停顿检测结果
- `metadata/clip_plan.json`
  - 结构化剪辑方案
- `metadata/clip_plan.csv`
  - 表格版剪辑方案
- `metadata/clip_plan.md`
  - 人工可读方案
- `subtitles/source.srt` / `source.vtt`
  - 整条视频字幕
- `subtitles/<clip_id>.srt` / `<clip_id>.vtt`
  - 每个 clip 的相对时间字幕
- `clips/<clip_id>.mp4`
  - 实际导出的视频片段

## 一个典型的数据流

```text
source video
  -> ffprobe
  -> extract audio
  -> mlx_whisper transcript
  -> ffmpeg silencedetect
  -> DeepSeek semantic clip selection
     or fallback local grouping
  -> pause-aware boundary snapping
  -> clip plan artifacts
  -> ffmpeg cut
  -> clip subtitles
```

## 当前策略的优点

- 对新闻节目很实用，尤其是主播串联结构明显的素材
- 本地转录，不依赖 Whisper 云 API
- 有停顿感知，边界比纯 transcript 分组自然
- DeepSeek 失败时不会整条流程中断
- 可复用 transcript、pause 和 clip plan，重跑成本低

## 当前策略的局限

- 不是按镜头切，没有接 `PySceneDetect`
- 没有人物 diarization，分不出主播、记者、采访对象的正式角色标签
- DeepSeek 和 fallback 是两套风格，clip 数量可能差很多
- 没有 clip budget 约束，同样 20 多分钟新闻，可能切成 10 几段，也可能切成 30 多段
- `DeepSeek` 返回结果目前只做基础校验，没有强力的后处理合并策略

## 测试

运行全部测试：

```bash
pytest -q
```

常用聚焦测试：

```bash
pytest tests/test_analyze_clips.py -q
pytest tests/test_build_clip_plan.py -q
pytest tests/test_run_pipeline.py -q
```

## 后续可增强方向

- 加入 `clip budget`，按节目总时长控制目标片段数
- 增加 post-pass merge，合并过碎的连续片段
- 接入 `PySceneDetect` 做镜头边界辅助
- 加入 speaker diarization，真正区分主播 / 记者 / 采访对象
- 增加缩略图和字幕烧录
- 支持更多视频类型，而不只偏新闻

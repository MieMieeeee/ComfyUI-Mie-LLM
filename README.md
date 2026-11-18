# ComfyUI-Mie-LLM

**Mie's LLM client layer.** Every provider connector is an HTTP client for
a chat endpoint; this package never loads a model itself, which is what
keeps the dependency surface at `requests`.

```bash
pip install -e <this directory>
```

It is the bottom layer of the three, and it depends on nothing of ours.

---

## What is in it

| Module | |
|---|---|
| `client.py` | the connector per provider - SiliconFlow, ZhiPu, Kimi, DeepSeek, MiniMax, MiMo, Doubao, Qianfan, Spark, OpenAI, Grok, OpenRouter, Claude, Gemini, Ollama, and the Jev decision connector |
| `budget.py` | the shared invoke, with the token-budget escalation for the prompt generators |
| `utils.py` | the helpers: the plugin config, the token resolution, the hash |
| `media.py` | the image helpers - the only part that needs numpy and opencv, and it imports them lazily, inside the functions that use them |

Importing `mie_llm.media` therefore costs nothing: a consumer that never
encodes an image never needs the heavy stack. The image helpers themselves
need `numpy` and `opencv-python-headless`, which are not dependencies of
this package - install them alongside it if you call them.

## Licence

GPL-3.0.


## 发布 / Releasing

```
python -m build
python -m twine upload dist/*
python tools/notify_mirrors.py --wait 300     # 让国内 PyPI 镜像收录
```

**第三步是发布流程的一部分, 不是可选项。** 国内 PyPI 镜像 (阿里云 / 华为云 / 清华)
对 PyPI 是镜像同步, 新版本通常要等它们自己的同步周期才出现; 期间按
`requirements.txt` 里的版本号去装会拿到 `No matching distribution found`。
这些镜像**有按需回源机制** —— 请求一次该包的 simple index 就能触发它们去 PyPI
拉取, 约 1 分钟内收录 (2026-11-18 实测: 请求后 61 秒华为云就从 0.1.x 变成
0.2.0)。所以发布后跑一次 `tools/notify_mirrors.py`, 把等待从「同步周期」压到
「约一分钟」。

## 依赖声明的规矩

**包只声明依赖, 不指定用哪个镜像源。** `requirements.txt` / `pyproject.toml` 里
只写包名和版本约束, **不写 `-i` / `--index-url` / `extra-index-url`** —— 镜像
是**部署环境**的选择 (启动器那边按用户在「启动与更新页」配的来), 包管不了也不该管。
写出镜像源会导致: 换环境装不上、用户没法用官方源、镜像滞后时把包锁死在旧版本。

同理, 装包的那一侧 (启动器的伴生服务安装链) 负责按配置选镜像, 镜像上没有的
版本由它回退官方 PyPI —— 不需要包自己操心。

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

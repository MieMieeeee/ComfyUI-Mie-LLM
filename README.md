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
| `media.py` | the image helpers - the only part that needs numpy, opencv, torch and pillow, and it is imported lazily |

## Licence

GPL-3.0.

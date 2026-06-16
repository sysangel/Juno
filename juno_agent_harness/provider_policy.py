"""Shared OpenRouter routing/privacy policy.

Keep policy-critical provider allow/deny lists here so the autonomous loop and
CLI harness use the same non-CN, no-retention OpenRouter routing screen.
"""
from __future__ import annotations

DEFAULT_CODE_MODEL_OPENROUTER = "deepseek/deepseek-v4-pro"
DEFAULT_TEST_MODEL_OPENROUTER = "deepseek/deepseek-v4-pro"
DEFAULT_MODEL_OPENROUTER = DEFAULT_CODE_MODEL_OPENROUTER
DEFAULT_BEDROCK_MODEL_ID = "us.deepseek.r1-v1:0"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REQUEST_TIMEOUT = 600
OPENROUTER_MAX_RETRIES = 1
OPENROUTER_REASONING_MAX_TOKENS = 3000
OPENROUTER_MAX_OUTPUT_TOKENS = 16000
OPENROUTER_CN_PROVIDERS = [
    "Baidu", "DeepSeek", "Moonshot AI", "Moonshot", "Alibaba", "Alibaba Cloud",
    "Qwen", "Zhipu", "Zhipu AI", "Z.AI", "ByteDance", "Volcengine", "Tencent",
    "Hunyuan", "MiniMax", "StepFun", "01.AI", "SiliconFlow", "iFlytek",
    "StreamLake", "Kuaishou", "SenseTime", "Baichuan", "InternLM",
]
OPENROUTER_WESTERN_PROVIDERS = [
    "DeepInfra", "Together", "Fireworks", "GMICloud", "Baseten",
    "Lambda", "Hyperbolic", "Nebius", "Parasail",
]
OPENROUTER_PROVIDER_PREFS = {
    "data_collection": "deny",
    "only": OPENROUTER_WESTERN_PROVIDERS,
    "ignore": OPENROUTER_CN_PROVIDERS,
    "sort": "price",
}
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
    "X-Title": "Self-Iterating Agent Loop",
}

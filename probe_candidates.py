"""Feasibility + privacy check for candidate models under the Western allowlist."""
from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

# Replicate agent_loop's privacy screen exactly.
CN_PROVIDERS = [
    "Baidu", "DeepSeek", "Moonshot AI", "Moonshot", "Alibaba", "Alibaba Cloud",
    "Qwen", "Zhipu", "Zhipu AI", "Z.AI", "ByteDance", "Volcengine", "Tencent",
    "Hunyuan", "MiniMax", "StepFun", "01.AI", "SiliconFlow", "iFlytek",
    "StreamLake", "Kuaishou", "SenseTime", "Baichuan", "InternLM",
]
WESTERN_PROVIDERS = [
    "DeepInfra", "Together", "Fireworks", "GMICloud", "Baseten",
    "Lambda", "Hyperbolic", "Nebius", "Parasail",
]
SCREEN = {
    "data_collection": "deny",
    "only": WESTERN_PROVIDERS,
    "ignore": CN_PROVIDERS,
    "sort": "price",
}
CN_LOWER = {p.lower() for p in CN_PROVIDERS}


def main() -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("NO_KEY_IN_ENV")
        return 2

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/angelsystems/agent-loop",
        "X-Title": "agent-loop-candidate-probe",
    }

    models = [
        "deepseek/deepseek-v4-pro",
        "minimax/minimax-m3",
        "qwen/qwen3.6-35b-a3b",
        "z-ai/glm-5.1",
        "moonshotai/kimi-k2.7-code",
    ]

    print("screen =", json.dumps(SCREEN))
    all_ok = True
    for model in models:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single token: ok"}],
            "max_tokens": 5,
            "temperature": 0,
            "provider": SCREEN,
        }
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=body, timeout=90,
        )
        d = r.json()
        served = d.get("provider")
        served_model = d.get("model")
        usage = d.get("usage")
        cn_flag = bool(served and str(served).lower() in CN_LOWER)

        print(f"\n{model}")
        print(f"  HTTP {r.status_code}  provider={served!r}  served_model={served_model!r}")
        print(f"  usage={json.dumps(usage)}  cn_flag={cn_flag}")
        if r.status_code != 200:
            print(f"  ERROR: {d.get('error', {}).get('message', json.dumps(d)[:300])}")
            all_ok = False
        elif cn_flag:
            print("  FAIL: served by CN provider")
            all_ok = False
    print("\n==============================")
    print("RESULT:", "ALL_OK" if all_ok else "SOME_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

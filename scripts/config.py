SAMPLING = dict(temperature=0.7, max_new_tokens=256)
EXP_MAX_NEW_TOKENS = {}

ALL_EXPS = ["exp1", "exp2", "exp3", "exp4"]
AUDIO_EXPS = ["exp1", "exp2"]

MODELS = {
    "qwen3-omni": dict(
        module="qwen3_omni", kind="local", env="vllm",
        hf="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        modalities={"audio", "image"}, exps=ALL_EXPS,
    ),
    "gemma-4-e4b": dict(
        module="gemma4", kind="local", env="gemma4",
        hf="google/gemma-4-E4B-it",
        modalities={"audio", "image"}, exps=ALL_EXPS,
    ),
    "gemini-3.5-flash": dict(
        module="gemini", kind="api", env="bk",
        hf=None, api_model="gemini-3.5-flash", api_key_env="GEMINI_API_KEY",
        modalities={"audio", "image"}, exps=ALL_EXPS,
    ),
    "minicpm-o-4.5": dict(
        module="minicpm_o", kind="local", env="vllm",
        hf="openbmb/MiniCPM-o-4_5",
        modalities={"audio", "image"}, exps=ALL_EXPS,
    ),
    "step-audio2": dict(
        module="step_audio2", kind="local", env="stepaudio2",
        hf="stepfun-ai/Step-Audio-2-mini",
        modalities={"audio"}, exps=AUDIO_EXPS,
    ),
    "audio-flamingo3": dict(
        module="audio_flamingo3", kind="local", env="af3",
        hf="nvidia/audio-flamingo-3-hf",
        modalities={"audio"}, exps=AUDIO_EXPS,
    ),
    "gpt-audio-1.5": dict(
        module="gpt_audio", kind="api", env="bk",
        hf=None, api_model="gpt-audio-1.5", api_key_env="OPENAI_API_KEY",
        modalities={"audio"}, exps=AUDIO_EXPS,
    ),
    "kimi-audio": dict(
        module="kimi_audio", kind="local", env="kimiaudio",
        hf="moonshotai/Kimi-Audio-7B-Instruct",
        modalities={"audio"}, exps=AUDIO_EXPS,
    ),
}


def get(model_name):
    if model_name not in MODELS:
        raise KeyError(f"unknown model '{model_name}'. known: {list(MODELS)}")
    return MODELS[model_name]

# Model and asset license register

This is an audit template, not legal advice or a declaration that any exact asset is approved.
Record the exact file/revision because model code, weights, voices, and datasets can use different
licenses. Keep copies of license texts and attribution requirements with release materials.

| Component | Candidate | Exact revision / files | Code license | Weight/voice license | Attribution / restrictions | Status |
|---|---|---|---|---|---|---|
| English streaming ASR | Sherpa-ONNX-compatible English model | TBD | Verify runtime | Verify model card | Verify | Pending |
| French streaming ASR | Sherpa-ONNX-compatible French model | TBD | Verify runtime | Verify model card | Verify | Pending |
| English/French/Turkish/German/Spanish/Italian/Mandarin ASR | Qwen/Qwen3-ASR-0.6B (optional) | Resolved by `models/fetch_qwen_asr.py` | Apache-2.0 package | Apache-2.0 model card | Preserve Qwen attribution and license | Candidate |
| Multilingual→EN and EN→multilingual translation | Helsinki-NLP OPUS-MT | Exact revisions and file hashes resolved by `models/fetch_translation_models.py` | Apache-2.0 model cards | Apache-2.0 model cards | Manifest records each exact revision and attribution | Candidate |
| Qwen3-TTS MLX Audio runtime | `mlx-audio` | Version pinned by `models/install_qwen_tts_mlx_runtime.py`; installed without dependencies to preserve the shared Transformers 4.57.6 runtime | MIT | N/A | Preserve runtime license; validate this compatibility path when upgrading either runtime | Candidate |
| Qwen3-TTS 0.6B Base 8-bit | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | Exact commit and file hashes resolved by `models/fetch_qwen_tts_mlx.py` | Apache-2.0 | Apache-2.0 | Preserve Qwen and mlx-community attribution | Candidate |
| Generic Cartoon reference | Piper `en_US-libritts_r-medium` speaker 0 | 4.04 second, 22.05 kHz mono PCM WAV in `src/real_time_translation/assets/cartoon_reference.wav` | Piper project: GPL-3.0-or-later | LibriTTS-R: CC BY 4.0 | “LibriTTS-R, CC BY 4.0; reference generated with Piper `en_US-libritts_r-medium`.” This is generic and does not reproduce a named character. | Candidate; verify redistribution terms for distribution |
| Generic Agent reference | Pitch-shifted copy of the Cartoon reference | 4.04 second, 22.05 kHz mono PCM WAV in `src/real_time_translation/assets/agent_reference.wav` | Piper project: GPL-3.0-or-later | LibriTTS-R: CC BY 4.0 | Same attribution as the source reference | Candidate; verify redistribution terms for distribution |

Before adoption, record source URL, retrieval date, exact revision/hash, SPDX/license text, intended
use permission, attribution text, and redistribution obligations. A framework's license does not
grant rights to every model distributed for that framework.


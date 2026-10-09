# Model and asset license register

This is an audit template, not legal advice or a declaration that any exact asset is approved.
Record the exact file/revision because model code, weights, voices, and datasets can use different
licenses. Keep copies of license texts and attribution requirements with release materials.

| Component | Candidate | Exact revision / files | Code license | Weight/voice license | Attribution / restrictions | Status |
|---|---|---|---|---|---|---|
| English streaming ASR | Sherpa-ONNX-compatible English model | TBD | Verify runtime | Verify model card | Verify | Pending |
| French streaming ASR | Sherpa-ONNX-compatible French model | TBD | Verify runtime | Verify model card | Verify | Pending |
| English/French ASR | Qwen/Qwen3-ASR-0.6B (optional) | Resolved by `models/fetch_qwen_asr.py` | Apache-2.0 package | Apache-2.0 model card | Preserve Qwen attribution and license | Candidate |
| EN→FR translation | Helsinki-NLP OPUS-MT | TBD | Apache-2.0 listing | Apache-2.0 listing | Preserve notices; verify exact revision | Candidate |
| FR→EN translation | Helsinki-NLP OPUS-MT | TBD | Apache-2.0 listing | Apache-2.0 listing | Preserve notices; verify exact revision | Candidate |
| Piper runtime | OHF-Voice/piper1-gpl | 1.8.0 API candidate | GPL-3.0-or-later | N/A | Review distribution obligations | Candidate |
| English TTS | Local Piper voice | TBD | Piper runtime above | Verify exact voice card | Verify speaker/data terms | Pending |
| French TTS | Local Piper voice | TBD | Piper runtime above | Verify exact voice card | Verify speaker/data terms | Pending |

Before adoption, record source URL, retrieval date, exact revision/hash, SPDX/license text, intended
use permission, attribution text, and redistribution obligations. A framework's license does not
grant rights to every model distributed for that framework.


# MAIX A — RunPod Serverless

This repository packages the MAIX A sound engine as a RunPod Serverless queue worker.

## Locked MAIX A sound
- MusicGen Melody: `facebook/musicgen-melody`
- Temperature: `0.4`
- CFG: `4.0`
- Top-K: `250`
- Top-P: `0.0`
- Native MusicGen output rate: `32 kHz`
- No 44.1/48 kHz resampling
- AudioCraft `strategy="loudness"`
- AudioCraft `loudness_compressor=True`
- WAV output
- Audio generation only for the exact `SAMPLE` command

## Composer Maestro
`action="chat"` lets Maestro discuss the composition.
`action="sample"` + `command="SAMPLE"` compiles the whole supplied conversation into the final MusicGen prompt, then generates audio.

The client must send the conversation on every request. The Serverless worker is intentionally stateless.

## Request: chat
See `test_chat.json`.

## Request: generate
See `test_sample.json`.

Optional melody conditioning fields:
- `reference_audio_base64`
- `reference_filename` (`.wav` or `.mp3`)
- `reference_start`
- `reference_seconds`

## Output
The sample response contains `audio_base64`. Decode it client-side and save it as a `.wav`.

## Build
Build for RunPod's x86_64 platform:

```bash
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/maix-serverless:v1 .
docker push YOUR_DOCKERHUB_USER/maix-serverless:v1
```

Then create a RunPod Serverless Queue endpoint from that image.

## Suggested first endpoint settings
- Active workers: 0 while testing
- Max workers: 1
- GPUs per worker: 1
- FlashBoot: enabled
- Execution timeout: 600 seconds
- Minimum CUDA: 11.8 or newer
- Start by testing a 24 GB GPU class. If it is not enough for MusicGen + Qwen together, move to a 48 GB class.

## Important production note
This first Serverless build keeps Qwen/Ollama and MusicGen in one GPU worker because it preserves the current MAIX behavior and is easy to deploy.

For production traffic, the cheaper architecture is likely:
1. CPU/cheap service for Maestro chat.
2. GPU Serverless endpoint only when the user sends `SAMPLE`.

That prevents ordinary chat messages from waking an expensive GPU worker.

## Model licensing
Check the licenses of every model before commercial deployment. In particular, the MusicGen model weights used here have licensing terms separate from the AudioCraft source code.

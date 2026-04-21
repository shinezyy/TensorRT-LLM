# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark online serving throughput for VisualGen (image/video generation).

On the server side, run:
    trtllm-serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --extra_visual_gen_options <config.yaml>

On the client side, run:
    python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --backend openai-videos \
        --prompt "A cat playing in the park" \
        --num-prompts 5 \
        --size 480x832 \
        --num-frames 81 \
        --fps 16 \
        --num-inference-steps 50 \
        --max-concurrency 1 \
        --save-result
"""

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
import traceback
from argparse import ArgumentParser as FlexibleArgumentParser
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import aiohttp
import numpy as np
import yaml
from tqdm.asyncio import tqdm

from tensorrt_llm.bench.benchmark.visual_gen_utils import (
    VisualGenRequestOutput,
    VisualGenSampleRequest,
    build_visual_gen_result_dict,
    calculate_metrics,
    load_visual_gen_prompts,
    print_visual_gen_results,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class VisualGenRequestInput:
    """HTTP request payload for online (server) benchmarking."""

    prompt: str
    api_url: str
    model: str
    size: str = "auto"
    seconds: float = 4.0
    fps: int = 24
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    extra_body: Optional[dict] = None
    # Media persistence (video backend). When ``save_media_path`` is set,
    # ``_do_post`` writes the successful response body to that file
    # instead of reading-and-discarding it. ``output_format`` is forwarded
    # to the server in the payload so the saved body matches the desired
    # container (``avi`` / ``mp4`` / ``auto``).
    save_media_path: Optional[str] = None
    output_format: Optional[str] = None


def _build_payload_common(request_input: VisualGenRequestInput) -> dict:
    """Build common payload fields shared by image and video generation."""
    payload: dict[str, Any] = {
        "model": request_input.model,
        "prompt": request_input.prompt,
        "size": request_input.size,
    }
    if request_input.num_inference_steps is not None:
        payload["num_inference_steps"] = request_input.num_inference_steps
    if request_input.guidance_scale is not None:
        payload["guidance_scale"] = request_input.guidance_scale
    if request_input.negative_prompt is not None:
        payload["negative_prompt"] = request_input.negative_prompt
    if request_input.seed is not None:
        payload["seed"] = request_input.seed
    if request_input.output_format is not None:
        payload["output_format"] = request_input.output_format
    if request_input.extra_body:
        payload.update(request_input.extra_body)
    return payload


def _get_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'unused')}",
    }


async def _do_post(
    request_input: VisualGenRequestInput,
    payload: dict[str, Any],
    pbar: Optional[tqdm],
    session: Optional[aiohttp.ClientSession],
) -> VisualGenRequestOutput:
    """Execute HTTP POST, measure E2E latency, return output."""
    request_session = session or aiohttp.ClientSession(
        trust_env=True,
        timeout=AIOHTTP_TIMEOUT,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
    )

    output = VisualGenRequestOutput()
    st = time.perf_counter()
    try:
        async with request_session.post(
            url=request_input.api_url, json=payload, headers=_get_headers()
        ) as response:
            if response.status == 200:
                if request_input.save_media_path:
                    # Stream the body to disk so the sweep produces
                    # playable media (e.g. .avi for LTX2-T2V). The
                    # server-side file persistence is independent of
                    # this; we save the bytes that were actually
                    # returned to the client.
                    save_path = request_input.save_media_path
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    with open(save_path, "wb") as fh:
                        async for chunk in response.content.iter_chunked(1 << 20):
                            fh.write(chunk)
                else:
                    await response.read()
                output.success = True
                output.e2e_latency = time.perf_counter() - st
            else:
                body = await response.text()
                output.error = f"HTTP {response.status}: {body}"
                output.success = False
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__
    finally:
        if session is None:
            await request_session.close()

    if pbar:
        pbar.update(1)
    return output


async def async_request_image_generation(
    request_input: VisualGenRequestInput,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> VisualGenRequestOutput:
    """POST /v1/images/generations and measure E2E latency."""
    payload = _build_payload_common(request_input)
    payload["response_format"] = "b64_json"
    payload["n"] = 1
    return await _do_post(request_input, payload, pbar, session)


async def async_request_video_generation(
    request_input: VisualGenRequestInput,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> VisualGenRequestOutput:
    """POST /v1/videos/generations (sync endpoint) and measure E2E latency."""
    payload = _build_payload_common(request_input)
    payload["seconds"] = request_input.seconds
    payload["fps"] = request_input.fps
    return await _do_post(request_input, payload, pbar, session)


VISUAL_GEN_REQUEST_FUNCS = {
    "openai-images": async_request_image_generation,
    "openai-videos": async_request_video_generation,
}


async def get_request(
    input_requests: list[VisualGenSampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[VisualGenSampleRequest, None]:
    """Asynchronously generates requests at a specified rate with optional burstiness."""
    assert burstiness > 0, f"A positive burstiness factor is expected, but given {burstiness}."
    theta = 1.0 / (request_rate * burstiness)
    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            continue
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: list[VisualGenSampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    selected_percentiles: list[float],
    max_concurrency: Optional[int],
    gen_params: dict[str, Any],
    extra_body: Optional[dict],
    no_test_input: bool = False,
    request_timeout: float = 6 * 60 * 60,
    num_gpus: int = 1,
    save_media_dir: Optional[str] = None,
    media_output_format: Optional[str] = None,
) -> dict[str, Any]:
    if backend not in VISUAL_GEN_REQUEST_FUNCS:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(VISUAL_GEN_REQUEST_FUNCS.keys())}"
        )

    request_func = VISUAL_GEN_REQUEST_FUNCS[backend]

    # Resolve the on-disk extension to use for saved media. The server
    # currently supports ``avi`` (default ext ``.avi``), ``mp4`` (ext
    # ``.mp4``), and ``auto`` (server picks). We echo the requested
    # format into the filename extension so external tools can identify
    # the container without running ``ffprobe``.
    ext_for_format = {"avi": ".avi", "mp4": ".mp4", "auto": ".bin"}
    save_ext = ext_for_format.get(media_output_format or "", ".bin")
    # Image backend bodies are JSON (base64-encoded PNGs); saving the
    # body as-is would be a JSON file, which is less useful. Keep the
    # feature video-only for now.
    save_media_for_backend = backend == "openai-videos"

    def _make_request_input(prompt: str, save_path: Optional[str] = None) -> VisualGenRequestInput:
        return VisualGenRequestInput(
            prompt=prompt,
            api_url=api_url,
            model=model_id,
            size=gen_params.get("size", "auto"),
            seconds=gen_params.get("seconds", 4.0),
            fps=gen_params.get("fps", 24),
            num_inference_steps=gen_params.get("num_inference_steps"),
            guidance_scale=gen_params.get("guidance_scale"),
            negative_prompt=gen_params.get("negative_prompt"),
            seed=gen_params.get("seed"),
            extra_body=extra_body,
            save_media_path=save_path,
            output_format=media_output_format if save_media_for_backend else None,
        )

    if not no_test_input:
        print("Starting initial single prompt test run...")
        test_input = _make_request_input(input_requests[0].prompt)
        test_output = await request_func(request_input=test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark "
                "arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        else:
            print("Initial test run completed. Starting main benchmark run...")
    else:
        print("Skipping initial test run. Starting main benchmark run...")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests), desc="Benchmarking")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(req_input, pbar_ref, sess):
        if semaphore is None:
            return await request_func(request_input=req_input, pbar=pbar_ref, session=sess)
        async with semaphore:
            return await request_func(request_input=req_input, pbar=pbar_ref, session=sess)

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
    ) as session:
        req_idx = 0
        async for request in get_request(input_requests, request_rate, burstiness):
            save_path: Optional[str] = None
            if save_media_dir and save_media_for_backend:
                save_path = os.path.join(
                    save_media_dir, f"sample_{req_idx:05d}{save_ext}"
                )
            request_input = _make_request_input(request.prompt, save_path=save_path)
            tasks.append(asyncio.create_task(limited_request_func(request_input, pbar, session)))
            req_idx += 1

        outputs: list[VisualGenRequestOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        selected_percentiles=selected_percentiles,
        num_gpus=num_gpus,
    )

    print_visual_gen_results(backend, model_id, benchmark_duration, metrics)

    result = build_visual_gen_result_dict(
        backend=backend,
        model_id=model_id,
        benchmark_duration=benchmark_duration,
        metrics=metrics,
        outputs=outputs,
        gen_params=gen_params,
    )

    return result


def load_prompts(args: argparse.Namespace) -> list[VisualGenSampleRequest]:
    """Load prompts from --prompt or --prompt-file (delegates to shared util)."""
    return load_visual_gen_prompts(args.prompt, args.prompt_file, args.num_prompts)


def _resolve_num_gpus(args: argparse.Namespace) -> int:
    """Determine the number of GPUs from explicit arg or server config YAML.

    Priority: --num-gpus (explicit) > --extra-visual-gen-options YAML > default 1.
    """
    if args.num_gpus is not None:
        return args.num_gpus

    if args.extra_visual_gen_options is not None:
        with open(args.extra_visual_gen_options, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        from tensorrt_llm.commands.utils import get_visual_gen_num_gpus

        return get_visual_gen_num_gpus(config)

    return 1


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model

    endpoint_map = {
        "openai-images": "/v1/images/generations",
        "openai-videos": "/v1/videos/generations",
    }
    endpoint = args.endpoint or endpoint_map.get(backend)
    if endpoint is None:
        raise ValueError(
            f"Cannot resolve endpoint for backend '{backend}'. "
            "Please specify --endpoint explicitly."
        )

    if args.base_url is not None:
        api_url = f"{args.base_url}{endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{endpoint}"

    input_requests = load_prompts(args)

    seconds = args.seconds
    if args.num_frames is not None:
        seconds = args.num_frames / args.fps
        print(f"Computed seconds={seconds:.3f} from num_frames={args.num_frames} / fps={args.fps}")

    gen_params: dict[str, Any] = {
        "size": args.size,
        "seconds": seconds,
        "fps": args.fps,
    }
    if args.num_inference_steps is not None:
        gen_params["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        gen_params["guidance_scale"] = args.guidance_scale
    if args.negative_prompt is not None:
        gen_params["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        gen_params["seed"] = args.seed

    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --extra-body: {e}") from e

    num_gpus = _resolve_num_gpus(args)

    gc.disable()

    # Resolve the media save directory. If ``--save-media-dir`` is not
    # set but ``--save-media`` is, fall back to ``--result-dir`` /
    # ``media`` so videos land next to the benchmark JSON.
    save_media_dir: Optional[str] = args.save_media_dir
    if args.save_media and save_media_dir is None:
        save_media_dir = os.path.join(args.result_dir or ".", "media")

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            max_concurrency=args.max_concurrency,
            gen_params=gen_params,
            extra_body=extra_body,
            no_test_input=args.no_test_input,
            request_timeout=args.request_timeout,
            num_gpus=num_gpus,
            save_media_dir=save_media_dir,
            media_output_format=args.media_output_format,
        )
    )

    if args.save_result:
        result_json: dict[str, Any] = {}

        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["num_prompts"] = args.num_prompts

        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    key, value = item.split("=", 1)
                    result_json[key.strip()] = value.strip()
                else:
                    raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")

        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            for field_name in ["e2e_latencies", "errors"]:
                result_json.pop(field_name, None)

        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency
        result_json["num_gpus"] = num_gpus

        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        )
        file_name = (
            f"{backend}-{args.request_rate}qps"
            f"{max_concurrency_str}-{base_model_id}"
            f"-{current_dt}.json"
        )
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(result_json, outfile, indent=2)

        print(f"Results saved to: {file_name}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark VisualGen (image/video generation) serving."
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="openai-videos",
        choices=list(VISUAL_GEN_REQUEST_FUNCS.keys()),
        help="Backend API type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Wan-AI/Wan2.1-T2V-14B).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--base-url", type=str, default=None, help="Full base URL (overrides --host/--port)."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="API endpoint path (auto-resolved from backend if not specified).",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single text prompt (repeated --num-prompts times).",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt file. Supports plain text (one prompt "
        "per line) or JSONL with 'text'/'prompt' field.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to benchmark."
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--size",
        type=str,
        default="auto",
        help="Output resolution in WxH format (e.g. 480x832) or 'auto'.",
    )
    gen_group.add_argument("--seconds", type=float, default=4.0, help="Video duration in seconds.")
    gen_group.add_argument("--fps", type=int, default=16, help="Frames per second.")
    gen_group.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Total frames to generate. Overrides --seconds (computed as num_frames / fps).",
    )
    gen_group.add_argument(
        "--num-inference-steps", type=int, default=None, help="Number of diffusion denoising steps."
    )
    gen_group.add_argument(
        "--guidance-scale", type=float, default=None, help="Classifier-free guidance scale."
    )
    gen_group.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    gen_group.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt (concepts to avoid)."
    )
    gen_group.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="JSON string of extra request body parameters (e.g. '{\"guidance_rescale\": 0.7}').",
    )

    traffic_group = parser.add_argument_group("Traffic Control")
    traffic_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (req/s). Default inf sends all at once.",
    )
    traffic_group.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request generation. 1.0 = Poisson process.",
    )
    traffic_group.add_argument(
        "--max-concurrency", type=int, default=None, help="Maximum concurrent requests."
    )
    traffic_group.add_argument(
        "--request-timeout",
        type=float,
        default=6 * 60 * 60,
        help="Request timeout in seconds (default: 6 hours).",
    )

    parser.add_argument(
        "--extra-visual-gen-options",
        type=str,
        default=None,
        help="Path to the server config YAML (same file passed to trtllm-serve "
        "via --extra_visual_gen_options). Parallelism settings are read to "
        "automatically determine the number of GPUs.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs used by the server. Overrides the value inferred "
        "from --extra-visual-gen-options. Defaults to 1 if neither is given.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-result", action="store_true", help="Save results to JSON file."
    )
    output_group.add_argument(
        "--save-detailed", action="store_true", help="Include per-request details in saved results."
    )
    output_group.add_argument(
        "--result-dir", type=str, default=None, help="Directory for result files."
    )
    output_group.add_argument(
        "--result-filename", type=str, default=None, help="Custom result filename."
    )
    output_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,99",
        help="Comma-separated percentile values (default: '50,90,99').",
    )
    output_group.add_argument(
        "--save-media",
        action="store_true",
        help="Persist each successful /v1/videos/generations response body "
        "to disk. Without --save-media-dir, files land under "
        "<result-dir>/media/. Video backend only.",
    )
    output_group.add_argument(
        "--save-media-dir",
        type=str,
        default=None,
        help="Explicit directory for saved media. Implies --save-media.",
    )
    output_group.add_argument(
        "--media-output-format",
        type=str,
        default=None,
        choices=[None, "avi", "mp4", "auto"],
        help="Value passed to the server's output_format field, and used "
        "as the saved-file extension. Default: server default.",
    )
    output_group.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=None,
        help="Key=value pairs to add to result metadata.",
    )

    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    parser.add_argument(
        "--no-test-input", action="store_true", help="Skip the initial single-prompt test run."
    )

    args = parser.parse_args()

    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file must be specified.")

    # --save-media-dir implies --save-media
    if args.save_media_dir and not args.save_media:
        args.save_media = True

    main(args)

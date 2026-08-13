import os
import json
import shutil
import tempfile
import importlib.util
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import logging

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)

# Tasks that use generate_until (benefit hugely from vLLM + PagedAttention)
GENERATIVE_TASKS = {"gsm8k", "svamp"}

# Alpaca Eval is handled by the `alpaca_eval` library (generation + LLM judge),
# NOT by lm_eval. Use this name in `--eval_tasks` to trigger it.
ALPACA_EVAL_TASK = "alpaca_eval"

try:
    from lm_eval.models.vllm_causallms import VLLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

# The lm_eval wrapper above imports even when the real `vllm` package is not
# installed (its module guards the import). For direct text generation we need
# to know whether the underlying vLLM engine is actually usable.
_VLLM_REALLY_AVAILABLE = importlib.util.find_spec("vllm") is not None


def _is_accelerate_dispatched(model):
    """Return True when the model is loaded with accelerate device dispatch/offload."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False

    if not isinstance(device_map, dict):
        return True

    mapped_devices = {str(v) for v in device_map.values()}
    if {"cpu", "disk", "meta"} & mapped_devices:
        return True
    return len(mapped_devices) > 1


def _safe_move_model(model, device):
    """Move model only when legal; skip for accelerate-dispatched/offloaded models."""
    if _is_accelerate_dispatched(model):
        log.info("Model uses accelerate dispatch/offload; skipping explicit model.to()/cpu() moves.")
        return model
    try:
        return model.to(device)
    except Exception as e:
        log.warning(f"Could not move model to {device}: {e}")
        return model


def _safe_empty_cuda_cache(context):
    """Best-effort CUDA cache cleanup that never raises."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"Skipping torch.cuda.empty_cache() after {context}: {e}")


def _model_supports_system_role(tokenizer):
    """Check if the model's chat template supports the system role."""
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def _run_hflm_eval(model, tokenizer, tasks, system_instruction, device):
    """Run evaluation using HuggingFace LM backend."""
    hflm_model = None
    try:
        # Avoid auto batch-size probing, which can be fragile after heavy GPU kernels.
        hflm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, dtype="bfloat16")
        results = simple_evaluate(
            model=hflm_model,
            tasks=tasks,
            apply_chat_template=True,
            system_instruction=system_instruction,
            batch_size=1,
            check_integrity=False,
            device=device,
        )
        return results
    except Exception as e:
        log.warning(f"HFLM evaluation failed: {e}. Returning no results for tasks: {tasks}")
        _safe_empty_cuda_cache("HFLM evaluation failure")
        return None
    finally:
        if hflm_model is not None:
            del hflm_model


def _run_vllm_eval(model, tokenizer, tasks, system_instruction, device):
    """
    Save the compressed model to a tmpdir and evaluate generative tasks
    via vLLM (PagedAttention + continuous batching).
    Falls back to HFLM loaded from the same tmpdir if vLLM fails — this
    avoids running inference on the in-memory quantized model whose Linear
    layers no longer expose a plain `.weight` attribute after compression.
    """
    tmpdir = tempfile.mkdtemp(prefix="prun_vllm_")
    try:
        log.info(f"Saving compressed model to {tmpdir} for vLLM...")
        model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)

        # Free some memory before vLLM loads weights from disk.
        # Do not call model.cpu()/model.to() when accelerate dispatch is active.
        _safe_empty_cuda_cache("saving model before vLLM load")

        log.info(f"Loading model in vLLM for generative tasks: {tasks}")
        vllm_model = VLLM(
            pretrained=tmpdir,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            trust_remote_code=True,
            batch_size="auto",
            enable_thinking=False,
        )
        results = simple_evaluate(
            model=vllm_model,
            tasks=tasks,
            apply_chat_template=True,
            system_instruction=system_instruction,
            check_integrity=False,
        )
        del vllm_model
        _safe_empty_cuda_cache("vLLM evaluation")
        return results
    except Exception as e:
        log.warning(f"vLLM evaluation failed: {e}. Falling back to HFLM from saved checkpoint.")
        _safe_empty_cuda_cache("vLLM failure")
        # Load from the saved tmpdir — do NOT pass the in-memory quantized
        # model because compressed-tensors replaces .weight with quantized
        # tensors that break vanilla F.linear after a cpu/gpu transfer.
        try:
            hflm_model = HFLM(
                pretrained=tmpdir,
                tokenizer=tokenizer,
                dtype="bfloat16",
                batch_size=1,
            )
            results = simple_evaluate(
                model=hflm_model,
                tasks=tasks,
                apply_chat_template=True,
                system_instruction=system_instruction,
                batch_size=1,
                check_integrity=False,
            )
            del hflm_model
            _safe_empty_cuda_cache("HFLM fallback")
            return results
        except Exception as e2:
            log.warning(f"HFLM fallback also failed: {e2}. Giving up on generative tasks.")
            _safe_empty_cuda_cache("HFLM fallback failure")
            return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _get_alpaca_eval_data():
    """Load the Alpaca Eval instructions + reference (GPT-4 baseline) outputs.

    Uses `hf_hub_download` to fetch the JSON directly, bypassing the deprecated
    datasets loading-script path that breaks on `datasets >= 4.0`
    (``tatsu-lab/alpaca_eval`` ships an ``alpaca_eval.py`` loader script).
    """
    from huggingface_hub import hf_hub_download

    ref_path = hf_hub_download(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline.json", repo_type="dataset"
    )
    with open(ref_path) as f:
        return json.load(f)


def _build_alpaca_eval_prompts(eval_data, tokenizer, system_instruction, supports_system):
    """Format each Alpaca Eval instruction as a chat prompt ready for generation."""
    prompts = []
    for item in eval_data:
        instruction = item["instruction"]
        input_text = item.get("input", "") or ""
        content = f"{instruction}\n\n{input_text}" if input_text else instruction

        messages = []
        if supports_system and system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def _generate_vllm_responses(model_dir, prompts, max_new_tokens=1024):
    """Generate long-form responses with a native vLLM engine from a saved checkpoint."""
    from vllm import LLM, SamplingParams

    vllm_engine = LLM(
        model=model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    outputs = vllm_engine.generate(prompts, sampling_params)
    responses = [output.outputs[0].text.strip() for output in outputs]
    del vllm_engine
    _safe_empty_cuda_cache("vLLM Alpaca Eval generation")
    return responses


def _generate_transformers_responses(model_dir, tokenizer, prompts, device, max_new_tokens=1024):
    """Generate long-form responses with HuggingFace transformers from a saved checkpoint."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    responses = []
    batch_size = 16
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=pad_token_id,
            )
        for j, out in enumerate(generated):
            input_len = inputs["input_ids"][j].shape[0]
            responses.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())

    del model
    _safe_empty_cuda_cache("transformers Alpaca Eval generation")
    return responses


def _build_local_judge_config(judge_model):
    """Build an alpaca_eval annotator config that runs a LOCAL open-weights judge
    model via transformers (``huggingface_local_completions``). This requires
    neither an OpenAI API key nor a vLLM server.

    It reuses the same pairwise ranking prompt as the built-in vLLM Llama-3-70B
    judge and parses the judge's ranking output with ``ranking_parser``.
    """
    from alpaca_eval import constants

    prompt_path = os.path.join(
        constants.EVALUATORS_CONFIG_DIR,
        "alpaca_eval_vllm_llama3_70b_fn",
        "alpaca_eval_fn.txt",
    )
    return {
        "local_judge": {
            "prompt_template": prompt_path,
            "fn_completions": "huggingface_local_completions",
            "completions_kwargs": {
                "model_name": judge_model,
                "do_sample": False,
                "batch_size": 1,
                "max_new_tokens": 300,
                "model_kwargs": {
                    "device_map": "auto",
                    "torch_dtype": "bfloat16",
                    "trust_remote_code": True,
                },
            },
            "fn_completion_parser": "ranking_parser",
        }
    }


def _run_alpaca_eval(
    model,
    tokenizer,
    system_instruction,
    supports_system,
    annotator_config=None,
    judge_model=None,
    max_instances=None,
):
    """Generate model responses on the Alpaca Eval set and score them with an LLM judge.

    Returns ``{"win_rate": float, "length_controlled_winrate": float,
    "standard_error": float}`` or ``None`` on any failure (so the rest of the
    evaluation for other tasks can still proceed).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from alpaca_eval import evaluate
    except ImportError as e:
        log.warning(f"alpaca_eval library not available: {e}")
        return None

    if judge_model:
        log.info(f"Using local judge model (no API key needed): {judge_model}")
        annotator_config = _build_local_judge_config(judge_model)
    elif annotator_config is None:
        annotator_config = os.environ.get(
            "ALPACA_EVAL_ANNOTATORS_CONFIG", "weighted_alpaca_eval_gpt4_turbo"
        )

    # Warn early if the chosen judge requires an OpenAI API key but none is set.
    # alpaca_eval's default (and the GPT-4-based configs) need OPENAI_API_KEY.
    if not judge_model and isinstance(annotator_config, str) and "gpt4" in annotator_config.lower():
        if not os.environ.get("OPENAI_API_KEY"):
            log.warning(
                f"Alpaca Eval judge config '{annotator_config}' requires an OpenAI "
                "API key, but OPENAI_API_KEY is not set. This will likely fail. "
                "Set --alpaca_eval_judge_model to a local model (e.g. "
                "meta-llama/Meta-Llama-3-8B-Instruct) to evaluate without an API key."
            )

    # Load the Alpaca Eval instructions + reference outputs.
    try:
        eval_data = _get_alpaca_eval_data()
    except Exception as e:
        log.warning(f"Could not load Alpaca Eval data: {e}")
        return None
    if max_instances is not None:
        eval_data = eval_data[:max_instances]
    log.info(f"Alpaca Eval: {len(eval_data)} instructions loaded.")

    # Save the (possibly compressed) model to a tmpdir, then generate from disk.
    # This mirrors _run_vllm_eval: compressed-tensors models don't expose plain
    # `.weight` after a cpu/gpu transfer, so we always re-load from disk.
    tmpdir = tempfile.mkdtemp(prefix="prun_alpaca_eval_")
    try:
        log.info(f"Saving model for Alpaca Eval generation to {tmpdir}...")
        model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        _safe_empty_cuda_cache("saving model for Alpaca Eval")

        prompts = _build_alpaca_eval_prompts(
            eval_data, tokenizer, system_instruction, supports_system
        )

        if _VLLM_REALLY_AVAILABLE:
            try:
                log.info("Generating Alpaca Eval responses with vLLM...")
                responses = _generate_vllm_responses(tmpdir, prompts)
            except Exception as e:
                log.warning(f"vLLM generation failed: {e}. Falling back to transformers...")
                _safe_empty_cuda_cache("vLLM Alpaca Eval generation failure")
                responses = _generate_transformers_responses(tmpdir, tokenizer, prompts, device)
        else:
            log.info("vLLM not available; generating Alpaca Eval responses with transformers...")
            responses = _generate_transformers_responses(tmpdir, tokenizer, prompts, device)

        model_outputs = [
            {"instruction": item["instruction"], "output": resp}
            for item, resp in zip(eval_data, responses)
        ]

        # If using a local judge (dict config), write it to a YAML file so that
        # alpaca_eval's Annotator.__init__ can load it via load_configs. A raw
        # dict would crash in _initialize_annotators_config (alpaca_eval v0.6.x
        # only handles paths, lists, or tuples there).
        if isinstance(annotator_config, dict):
            import yaml
            config_path = os.path.join(tmpdir, "local_judge_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(annotator_config, f)
            annotator_config = config_path

        # Run the judge (default: weighted GPT-4 Turbo via OpenAI API).
        log.info(f"Running Alpaca Eval judge (annotator: {annotator_config})...")
        df_leaderboard, _ = evaluate(
            model_outputs=model_outputs,
            reference_outputs=eval_data,
            annotators_config=annotator_config,
            output_path=None,          # do not persist leaderboard / annotations
            precomputed_leaderboard=None,  # do not merge cached leaderboard rows
            is_return_instead_of_print=True,
        )

        row = df_leaderboard.iloc[0] if len(df_leaderboard) > 0 else {}
        metrics = {}
        for col in ["win_rate", "length_controlled_winrate", "standard_error"]:
            if col in row and row[col] is not None:
                metrics[col] = float(row[col])
        log.info(f"Alpaca Eval results: {metrics}")
        return metrics
    except Exception as e:
        log.warning(f"Alpaca Eval evaluation failed: {e}")
        _safe_empty_cuda_cache("Alpaca Eval failure")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



def evaluate_model(
    model_name,
    model,
    tokenizer,
    task_list=[],
    use_accelerate=False,
    add_special_tokens=False,
    alpaca_eval_annotator_config=None,
    alpaca_eval_judge_model=None,
    alpaca_eval_max_instances=None,
):
    """
    Evaluate a given model on specified tasks using the lm_eval framework.
    Generative tasks (gsm8k, svamp) use vLLM with PagedAttention when available.
    All other tasks use HFLM.
    The special task ``alpaca_eval`` is handled by the ``alpaca_eval`` library
    (generation + LLM judge) rather than lm_eval.

    Args:
        model_name (str): Name of the model.
        model (torch.nn.Module): The model to evaluate.
        tokenizer: Tokenizer for the model.
        task_list (list): Task names to evaluate on.
        use_accelerate (bool): Whether to use HuggingFace Accelerate.
        add_special_tokens (bool): Whether to add special tokens.
        alpaca_eval_annotator_config (str, optional): Name of the annotator
            (judge) config to use for Alpaca Eval. Defaults to the
            ``ALPACA_EVAL_ANNOTATORS_CONFIG`` env var, else
            ``weighted_alpaca_eval_gpt4_turbo``.
        alpaca_eval_judge_model (str, optional): A local open-weights judge model
            (path or HF id) to run in-process via transformers. This avoids the
            need for an OpenAI API key. Overrides ``alpaca_eval_annotator_config``.
        alpaca_eval_max_instances (int, optional): Cap the number of Alpaca Eval
            instructions evaluated (for quick smoke tests).
    Returns:
        dict: ``{"results": {task: {metric: value, ...}, ...}}``
    """
    log.info(f"Evaluating model {model_name} on tasks: {task_list}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure model is on the correct device when this is legal.
    try:
        if model.device.type != device:
            log.info(f"Moving model to {device}...")
            model = _safe_move_model(model, device)
    except AttributeError:
        pass  # sharded / device_map models

    log.info(f"Model device: {device}")

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    # Detect system-role support once
    supports_system = _model_supports_system_role(tokenizer)
    if supports_system:
        log.info("Model supports system role, using chat template with system instruction.")
        system_instruction = "You are a helpful assistant."
    else:
        log.info("Model does NOT support system role, disabling system instruction.")
        system_instruction = None

    # Separate the Alpaca Eval task (handled by the alpaca_eval library, not lm_eval).
    alpaca_eval_in_list = ALPACA_EVAL_TASK in task_list
    lm_eval_tasks = [t for t in task_list if t != ALPACA_EVAL_TASK]

    # Split tasks: generative (vLLM) vs loglikelihood (HFLM)
    gen_tasks = [t for t in lm_eval_tasks if t in GENERATIVE_TASKS]
    ll_tasks = [t for t in lm_eval_tasks if t not in GENERATIVE_TASKS]

    all_results = {}

    # 1. Run loglikelihood tasks with HFLM (fast already)
    if ll_tasks:
        log.info(f"Running loglikelihood tasks with HFLM: {ll_tasks}")
        ll_results = _run_hflm_eval(model, tokenizer, ll_tasks, system_instruction, device)
        if ll_results and "results" in ll_results:
            all_results.update(ll_results["results"])

    # 2. Run generative tasks with vLLM (PagedAttention) if available
    if gen_tasks:
        if _VLLM_AVAILABLE:
            log.info(f"Running generative tasks with vLLM (PagedAttention): {gen_tasks}")
            gen_results = _run_vllm_eval(model, tokenizer, gen_tasks, system_instruction, device)
        else:
            log.info(f"vLLM not available, using HFLM for generative tasks: {gen_tasks}")
            # Move model back to GPU if it was offloaded
            try:
                if model.device.type != device:
                    model = _safe_move_model(model, device)
            except AttributeError:
                pass
            gen_results = _run_hflm_eval(model, tokenizer, gen_tasks, system_instruction, device)

        if gen_results and "results" in gen_results:
            all_results.update(gen_results["results"])

    # 3. Run Alpaca Eval if requested (generation + LLM judge)
    if alpaca_eval_in_list:
        log.info("Running Alpaca Eval evaluation...")
        alpaca_metrics = _run_alpaca_eval(
            model,
            tokenizer,
            system_instruction,
            supports_system,
            annotator_config=alpaca_eval_annotator_config,
            judge_model=alpaca_eval_judge_model,
            max_instances=alpaca_eval_max_instances,
        )
        if alpaca_metrics:
            all_results[ALPACA_EVAL_TASK] = alpaca_metrics

    return {"results": all_results}

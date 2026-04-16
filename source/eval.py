import os
import tempfile
import shutil
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

try:
    from lm_eval.models.vllm_causallms import VLLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


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
    hflm_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", dtype="bfloat16")
    results = simple_evaluate(
        model=hflm_model,
        tasks=tasks,
        apply_chat_template=True,
        system_instruction=system_instruction,
        batch_size="auto",
        check_integrity=False,
        device=device,
    )
    del hflm_model
    return results


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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        log.warning(f"vLLM evaluation failed: {e}. Falling back to HFLM from saved checkpoint.")
        torch.cuda.empty_cache()
        # Load from the saved tmpdir — do NOT pass the in-memory quantized
        # model because compressed-tensors replaces .weight with quantized
        # tensors that break vanilla F.linear after a cpu/gpu transfer.
        try:
            hflm_model = HFLM(
                pretrained=tmpdir,
                tokenizer=tokenizer,
                dtype="bfloat16",
                batch_size="auto",
            )
            results = simple_evaluate(
                model=hflm_model,
                tasks=tasks,
                apply_chat_template=True,
                system_instruction=system_instruction,
                check_integrity=False,
            )
            del hflm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return results
        except Exception as e2:
            log.warning(f"HFLM fallback also failed: {e2}. Giving up on generative tasks.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
):
    """
    Evaluate a given model on specified tasks using the lm_eval framework.
    Generative tasks (gsm8k, svamp) use vLLM with PagedAttention when available.
    All other tasks use HFLM.

    Args:
        model_name (str): Name of the model.
        model (torch.nn.Module): The model to evaluate.
        tokenizer: Tokenizer for the model.
        task_list (list): Task names to evaluate on.
        use_accelerate (bool): Whether to use HuggingFace Accelerate.
        add_special_tokens (bool): Whether to add special tokens.
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

    # Split tasks: generative (vLLM) vs loglikelihood (HFLM)
    gen_tasks = [t for t in task_list if t in GENERATIVE_TASKS]
    ll_tasks = [t for t in task_list if t not in GENERATIVE_TASKS]

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

    return {"results": all_results}

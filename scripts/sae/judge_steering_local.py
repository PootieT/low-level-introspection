#!/usr/bin/env python3
import argparse, ast, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

RUBRIC = (
    "You judge whether short text continuations express a given concept.\n"
    "For each option, answer true only if BOTH hold:\n"
    "1. The option's main content clearly expresses the concept. Content that is only "
    "tangential, partial, or loosely related does not count.\n"
    "2. The option is coherent text: fluent, grammatical, meaningful sentences that are not "
    "repetitive. Broken, looping or repetitive text, or text that merely repeats or lists the "
    "concept's words instead of writing about it, does not count, even if it names the "
    "concept. The option may change topic away from the text it follows and may be factually "
    "implausible; that does not make it incoherent.\n"
    "Judge each option on its own text; do not compare options or guess how they were produced.\n"
    "Reply with JSON only, exactly: {REPLY}"
)


def rubric(letters):
    return RUBRIC.replace("{REPLY}", "{" + ", ".join(f'"{k}": true|false' for k in letters) + "}")


def parse_judge_md(path):
    md = Path(path).read_text()
    out = []
    for block in re.split(r"^## Feature ", md, flags=re.M)[1:]:
        fid = block.split("\n", 1)[0].strip()
        concept = re.search(r"\*\*Concept:\*\*\s*(.+)", block).group(1).strip()
        prompts = []
        for pb in re.split(r"^\*\*Prompt \d+:\*\* ", block, flags=re.M)[1:]:
            lines = pb.split("\n")
            opts = {}
            for ln in lines[1:]:
                m = re.match(r"- \*\*([A-Z])\.\*\* (.+)$", ln)
                if m:
                    opts[m.group(1)] = ast.literal_eval(m.group(2))
            assert sorted(opts) == [chr(65 + i) for i in range(len(opts))] and len(opts) >= 2, (fid, lines[0])
            prompts.append((lines[0].strip(), opts))
        out.append((fid, concept, prompts))
    return out


def samples_sha(feat):
    return hashlib.sha256(json.dumps([RUBRIC, *feat[1:]], sort_keys=True).encode()).hexdigest()[:16]


def user_message(concept, prompt, opts):
    return (f"Concept: {concept}\n\nText the continuations follow: {prompt}\n\n"
            + "\n\n".join(f"Option {k}: {opts[k]!r}" for k in sorted(opts)))


def parse_answer(text, letters):
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(isinstance(d.get(k), bool) for k in letters):
        return None
    return [k for k in letters if d[k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-md", type=Path, nargs="+", required=True)
    p.add_argument("--features", nargs="*", help="Only these feature ids (default: all in the files).")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--revision", default=None, help="Pin the model snapshot (commit hash).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=256,
                   help="Concurrent sequences. Qwen3.8 is hybrid (Mamba-style layers, one cache "
                        "block per sequence); vLLM's default 1024 exceeds the blocks that fit.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    feats = [f for path in args.judge_md for f in parse_judge_md(path)]
    if args.features:
        feats = [f for f in feats if f[0] in set(args.features)]
    (args.out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "picks").mkdir(parents=True, exist_ok=True)
    def done(f):
        p = args.out_dir / "picks" / f"{f[0]}.json"
        return p.exists() and json.loads(p.read_text()).get("samples_sha") == samples_sha(f)
    todo = [f for f in feats if not done(f)]
    print(f"[judge] {len(feats)} features, {len(todo)} not yet judged on their current samples")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, revision=args.revision, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=args.seed)
    kw = dict(chat_template_kwargs={"enable_thinking": False})

    conv, index = [], []
    for fid, concept, prompts in todo:
        for i, (pr, opts) in enumerate(prompts, 1):
            conv.append([{"role": "system", "content": rubric(sorted(opts))},
                         {"role": "user", "content": user_message(concept, pr, opts)}])
            index.append((fid, i, sorted(opts)))
    outs = llm.chat(conv, sp, use_tqdm=True, **kw)
    texts = [o.outputs[0].text for o in outs]

    bad = [j for j, t in enumerate(texts) if parse_answer(t, index[j][2]) is None]
    if bad:
        retry = [conv[j][:1] + [{"role": "user", "content": conv[j][1]["content"]
                                 + "\n\nAnswer with the JSON object only."}] for j in bad]
        for j, o in zip(bad, llm.chat(retry, sp, use_tqdm=False, **kw)):
            texts[j] = o.outputs[0].text
    concept_of = {fid: c for fid, c, _ in todo}
    sha_of = {f[0]: samples_sha(f) for f in todo}
    per = {}
    for (fid, i, letters), t in zip(index, texts):
        per.setdefault(fid, []).append((i, t, parse_answer(t, letters)))
    n_fail = 0
    for fid, rows in per.items():
        with open(args.out_dir / "raw" / f"{fid}.jsonl", "w") as fh:
            for i, t, a in rows:
                fh.write(json.dumps({"prompt": i, "response": t, "picks": a}) + "\n")
        picks = {str(i): a for i, t, a in rows if a is not None}
        bad = sum(a is None for _, _, a in rows)
        n_fail += bad
        if bad:
            print(f"[judge] {fid}: {bad} prompts unparsed after retry; see raw/{fid}.jsonl")
            continue
        (args.out_dir / "picks" / f"{fid}.json").write_text(
            json.dumps({"feature": fid, "concept": concept_of[fid], "samples_sha": sha_of[fid],
                        "picks": picks}))
    info = {"model": args.model, "revision_requested": args.revision, "rubric": RUBRIC, "temperature": 0.0, "max_tokens": 64,
            "enable_thinking": False, "seed": args.seed, "date": datetime.now(timezone.utc).isoformat(),
            "judge_md": [str(x) for x in args.judge_md], "unparsed_after_retry": n_fail}
    try:
        from huggingface_hub import try_to_load_from_cache
        info["revision"] = Path(try_to_load_from_cache(args.model, "config.json",
                                                       revision=args.revision)).parent.name
    except Exception as e:
        info["revision"] = f"unresolved ({e.__class__.__name__}); requested {args.revision}"
    prev = args.out_dir / "run_info.json"
    runs = json.loads(prev.read_text()) if prev.exists() else []
    prev.write_text(json.dumps(runs + [info], indent=1))
    print(f"[judge] wrote {len(per)} features; {n_fail} prompts unparsed after retry")
    if n_fail:
        sys.exit(f"[judge] {n_fail} prompts could not be parsed; those features have no picks")


if __name__ == "__main__":
    main()

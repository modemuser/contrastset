# Built-in prompt templates

A template is the exact text the subject model sees. `{{question}}` is replaced by
the question and `{{entity}}` by the entity name. Its SHA-256 hash is recorded in
every graded line.

Built-ins carry the **chat format only**: no answer instruction, no refusal phrases.
What a good answer looks like ("just the name", "just the number") and what a
refusal looks like depend on the task, so they belong in your own template file,
which may start with a front-matter block of refusal phrases:

    ---
    refusal_phrases: ["i don't know", "not sure"]
    ---
    <|im_start|>user
    {{question}} Answer with just the name. If you do not know, say "I don't know".<|im_end|>
    ...

The adapters send the text as-is, with no server-side chat template. Use the same
format the downstream probing experiment will use, or grading and probing measure
different prompts.

- `qwen3-nothink.txt`: Qwen3 ChatML (Qwen's chat markup) with an empty `<think>` block,
  which is what Qwen's own template emits for `enable_thinking=False`.

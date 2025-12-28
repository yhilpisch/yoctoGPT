# Agent Guidelines for this Repository

These instructions apply to the entire `yoctoGPT` project unless overridden
by a more specific `AGENTS.md` in a subdirectory.

## General Style

- Keep changes minimal and focused on the task at hand.
- Match the existing code and documentation style where possible.
- Prefer clear, readable code over clever but obscure constructs.

## Notebooks and Shell Commands (Must)

When editing or adding Jupyter/Colab notebooks (e.g. files under `notebooks/`):

- Long shell or magic commands (such as `!python -m ...`) **must**:
  - Use exactly **one space** between tokens/arguments (no repeated blanks).
  - Be wrapped over multiple lines when they would otherwise be long, using
    backslash line continuations, for example:

    ```bash
    !python -m scripts.recommend_training \
      --mode token \
      --data_dir data/token \
      --tokenizer_path data/token/tokenizer.json \
      --ckpt_dir {CKPT_DIR} \
      --priority speed \
      --device cuda \
      --device_mem_gb 12
    ```

- Apply the same wrapping rule to other long `!python`, `!pip`, or similar
  notebook commands so that they stay reasonably short (≈85 characters or less
  per line).
- Top-level lines in code cells (including `#@title` lines, imports, and
  shell/magic commands) must not have unnecessary leading indentation:
  start them at column 0 and only indent where Python syntax requires it
  (e.g. inside `if`, `for`, `with` blocks).

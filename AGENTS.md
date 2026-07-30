# Repository Guidelines

## Project Structure & Module Organization

The public C++ API is in `include/hunyuan_ocr/`; implementation code is in `src/`, with
private components in `src/internal/`. The CLI lives in `apps/`. Model-export and
validation utilities are under `tools/`. Native and pytest tests live in `tests/cpp/`
and `tests/python/`; fixtures are in `tests/data/`. Keep documentation in `docs/`.
Treat `third_party/` as vendored code and avoid unrelated edits there.

## Build, Test, and Development Commands

Initialize dependencies and run the standard release workflow (any other build flags are also allowed, depending on the specific task):

```bash
cmake --preset release
cmake --build --preset release -j
ctest --preset release
```

Run Python tests with `pytest -q tests/python`. Heavy OCR regressions require exported
files in `assets/hunyuan_ocr/` and are opt-in:

```bash
HUNYUAN_OCR_RUN_E2E=1 pytest -q tests/python/test_e2e.py
```

Check Python with `ruff check .` and `ruff format --check .`; format C++ with
`clang-format -i <files>`. See `README.md` for model export and Vulkan builds.

## Coding Style & Naming Conventions

Use four spaces (two for JSON/YAML), LF endings, and a final newline. C++ follows the
Google-based `.clang-format` profile with a 100-column limit. Use `snake_case` for
functions, variables, and files, and `PascalCase` for C++ types. Python targets 3.10
and follows `pyproject.toml` Ruff rules.

## Testing Guidelines

Add focused native coverage to `runtime_core_test.cpp` or a replay executable. Name
Python tests and functions `test_*`. Keep fixtures in `tests/data/`; do not silently
regenerate goldens. Explicitly gate tests needing models, downloads, or Vulkan.

## Development Progress Log

During development, append to `docs/private/progress-log.md` after each meaningful
implementation, validation, or benchmark milestone. Never rewrite earlier entries. Use
UTC and this format:

```markdown
## YYYY-MM-DD HH:MM UTC — <Task>: <Milestone>
- **Task:** Scope and intended outcome.
- **Result:** What was completed and its observable behavior.
- **Changes:** Important files, components, or artifacts changed.
- **Validation:** Exact commands/checks and pass, fail, or skipped status.
- **Metrics:** Core before/after values (tests, latency, memory, accuracy, size, etc.).
- **Follow-up:** Remaining risks or next action; use `None` when complete.
```

Keep metrics comparable by recording inputs and relevant settings such as backend,
precision, and thread count. Write `N/A — <reason>` when no meaningful metric exists;
do not omit the field. One entry should describe one completed, reviewable milestone.

## Commit & Pull Request Guidelines

Use concise imperative subjects and Conventional Commit prefixes when applicable, for
example `fix: preserve decoder cache shape`. Pull requests should explain behavior,
list validation commands, link issues, and disclose model, fixture, or performance
impacts. Include before/after metrics for inference changes.

---
name: cambricon-mlu-context
description: Local context and discovery guide for Cambricon / MLU work. Use when Codex is working on Cambricon-, MLU-, Neuware-, CNRT-, BANG C-, cnfatbin-, or mlir-mlu-compiler-related tasks and needs local documentation, demos, examples, or environment-specific references. This skill tells Codex to consult /home/olafxiong/workspace/mlu for relevant docs and demo code as needed.
---

# Cambricon Mlu Context

Use `/home/olafxiong/workspace/mlu` as the primary local reference root for Cambricon / MLU work.

## Workflow

1. Search `/home/olafxiong/workspace/mlu` before making assumptions about Cambricon-specific behavior, APIs, toolchain flags, runtime usage, or demo layouts.
2. Prefer targeted discovery with `rg` or `find` instead of loading large trees into context.
3. When the task mentions a specific tool, API, or artifact, search for that exact term first. Typical terms include `mlu`, `cambricon`, `neuware`, `cnrt`, `bang`, `fatbin`, `cnfatbin`, `mlir-mlu-compiler`, and relevant operator or pass names.
4. Read only the files directly relevant to the current task, then summarize the result and continue the work.
5. If local examples or docs in `/home/olafxiong/workspace/mlu` conflict with prior assumptions, follow the local materials.

## Quick Commands

- List files:
  `find /home/olafxiong/workspace/mlu -maxdepth 2 -type f | head`
- Search keywords:
  `rg -n "cambricon|mlu|neuware|cnrt|bang|cnfatbin|mlir-mlu-compiler" /home/olafxiong/workspace/mlu`
- Find demos for a feature:
  `rg -n "<keyword>" /home/olafxiong/workspace/mlu`

## Scope

This skill does not bundle copied references. It points Codex at the existing local source of truth in `/home/olafxiong/workspace/mlu`.

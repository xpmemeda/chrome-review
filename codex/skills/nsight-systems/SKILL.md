---
name: nsight-systems
description: 使用 NVIDIA Nsight Systems（nsys）进行性能采集和 .nsys-rep 报告分析。适用于选择或运行分析 recipe、通过 SQL 查询报告、核对 nsys CLI 语法及文档，以及排查 CUDA、图形、CPU/GPU 性能、卡顿、帧率下降、ETW 采集和 ETL 导入问题。

license: Proprietary
metadata:
  author: Joan Yi <jyi@nvidia.com>
  tags:
    - nvidia
    - nsys
    - developer-tools
    - profiling
    - performance
    - cuda
    - graphics
    - report-analysis
---

# Nsight Systems

## Workflow

### General steps

1. Run the appropriate script for your platform, with no arguments, to locate the nsys tool: `sh scripts/locate_nsys.sh` on Linux or `scripts\locate_nsys.bat` on Windows. These scripts are located in the scripts folder next to this `SKILL.md` file.
2. `<nsys>` means the exact Nsys CLI executable path printed by the locator. Use that path whenever this skill, CLI help, documentation, recipe output, or a follow-up suggestion tells you to run Nsys.
3. Resolve errors and missing dependencies (if any) before continuing.
4. Run `<nsys> --help` to learn about Nsys CLI commands and their purpose.
5. Run `<nsys> <command> --help` for the exact arguments of the command you picked.
6. Analyze the user's prompt and extract relevant keywords and topics and pass them to `<nsys> docs` in order to get definitive Nsys workflow. For each relevant result, open the returned documentation and read the complete suggested section. You MUST do this before you continue to the next steps as otherwise you risk misleading the user.

### Additional steps for workflows that involve nsys-rep files

#### First-Line Analysis
7. Run `<nsys> survey` to get a high-level view of the report and identify applicable recipes. If SQLite full output schema is needed, use the `--full-schema` flag.
8. Run `<nsys> recipe` when a recipe applies to the report and may provide useful analysis for the task. Run the most relevant applicable recipe first. Read the recipe's insights first and consider any follow-up guidance.

#### Analysis Assessment
9. Assess the recipe's evidence. If it introduces a new topic or analysis direction, search it with `<nsys> docs` and read the relevant full sections before continuing. You MUST do this, as otherwise you risk misleading the user.
10. Decide whether the evidence answers the user's question directly. Do not stop at a symptom or correlation.
11. If deeper analysis requires querying any previously run recipe's output, use `<nsys> query` on its Parquet files, selecting tables from the schema specified in the insights file.
12. If the evidence is not sufficient, and before starting Second-Line Analysis, run any other survey-listed recipe that may help gather the missing evidence.
13. ONLY if you've exhausted the first-line analysis directions, and the gathered evidence is insufficient, continue to the second-line analysis.

#### Second-Line Analysis
14. If no recipe fits, or applicable recipes and their outputs cannot provide the required evidence, run `<nsys> query` directly on the report. Select tables and columns from the schema specified by `<nsys> survey --full-schema`. Use `<nsys> export` or custom SQLite queries only when you cannot determine the report content from the previous steps.
15. Assess the query's evidence. If it introduces a new topic or analysis direction, search it with `<nsys> docs` and read the relevant full sections before continuing. You MUST do this, as otherwise you risk misleading the user.

#### Finalize Answer
16. Finalize an evidence-based answer by combining the gathered report evidence with documentation that explains the observed data, its limits, and supported conclusions.

## Guidelines

- Follow one analysis direction at a time. Wait for each command to finish and inspect its output before choosing the next step.
- Do not run competing recipes, exports, queries, or scripts in parallel. Change direction only when the current direction fails or cannot provide the required evidence.
- Distinguish per-report results from directory aggregates.
- Distinguish “not captured” from “captured but absent.” Missing data does not prove an event did not happen. Claim zero events only when the relevant data was both captured and absent.
- State only conclusions supported by observed command or report output. If required evidence is missing, state the limit; do not substitute another report or invent details.

## Response style

You are an NVIDIA Nsight Systems performance expert. Answer from evidence gathered in this session: exported report data for measured facts, live `nsys` help for CLI facts, live/packaged recipe metadata for recipe facts, and packaged Nsight Systems references for product workflows.
Be concise for report-data answers and show the evidence table or summary. Be more explanatory for workflow/docs answers.
Answer the question that was asked before adding adjacent operational advice.
No preambles or conversational fluff, and no running narration of your process — but state the method or recipe that produced a result, since the user may want to reproduce it. Answer simple requests directly; briefly restate a complex task first to confirm interpretation. Write declaratively at a professional level; skip hand-holding and generic advice. Use active voice, omit needless words, and state facts positively (what is true, not what is not). Prefer bullets for disjoint facts, numbered steps only for a strict sequence, and tables or short fragments over prose.

#!/usr/bin/env python3
# Copyright 2026 The Chrome Review Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate an Ocean BABI CSV and generate CSV and HTML reports."""

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


BILLING_TYPES = ("input", "input-cache", "output")
CSV_HEADERS = (
    "Customer",
    "Model",
    "Input (CNY)",
    "Input Cache (CNY)",
    "Output (CNY)",
    "Total (CNY)",
)


def ParseArguments() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Aggregate Ocean BABI costs by customer and model, then generate "
          "an English CSV and a formatted HTML report."
      )
  )
  parser.add_argument("input_csv", type=Path, help="Path to the source CSV.")
  parser.add_argument(
      "--output-csv",
      type=Path,
      help="Output CSV path. Defaults to <input>-by-customer.csv.",
  )
  parser.add_argument(
      "--output-html",
      type=Path,
      help="Output HTML path. Defaults to <input>-by-customer.html.",
  )
  parser.add_argument(
      "--title",
      default="Ocean BABI Cost Summary",
      help="Title displayed in the HTML report.",
  )
  parser.add_argument(
      "--top",
      type=int,
      default=10,
      help="Number of leading rows displayed separately in the pie chart.",
  )
  return parser.parse_args()


def ParseAmount(value: str) -> Decimal:
  """Converts a CSV amount to Decimal, treating an empty value as zero."""
  try:
    return Decimal(value or "0")
  except InvalidOperation as error:
    raise ValueError(f"Invalid amount: {value!r}") from error


def SplitCustomer(customer: str) -> Tuple[str, str, str]:
  """Splits Customer into business customer, model, and billing type."""
  parts = customer.rsplit("|", 2)
  if len(parts) == 3 and parts[2] in BILLING_TYPES:
    return parts[0], parts[1], parts[2]
  if len(parts) >= 2:
    # Some bundled model charges have no input/output suffix. Assign them to
    # output so that the report retains the complete source total.
    return "|".join(parts[:-1]), parts[-1], "output"
  raise ValueError(f"Cannot parse Customer value: {customer!r}")


def AggregateRows(input_csv: Path) -> List[Dict[str, str]]:
  """Aggregates CNY amounts by business customer, model, and billing type."""
  amounts = defaultdict(
      lambda: {billing_type: Decimal("0") for billing_type in BILLING_TYPES}
  )

  with input_csv.open(encoding="utf-8-sig", newline="") as source_file:
    reader = csv.DictReader(source_file)
    required_columns = {"Customer", "金额(CNY)"}
    missing_columns = required_columns.difference(reader.fieldnames or ())
    if missing_columns:
      missing_text = ", ".join(sorted(missing_columns))
      raise ValueError(f"Missing required columns: {missing_text}")

    for source_row in reader:
      customer, model, billing_type = SplitCustomer(source_row["Customer"])
      amounts[(customer, model)][billing_type] += ParseAmount(
          source_row["金额(CNY)"]
      )

  rows = []
  for (customer, model), values in amounts.items():
    total = sum(values.values(), Decimal("0"))
    rows.append(
        {
            "Customer": customer,
            "Model": model,
            "Input (CNY)": format(values["input"], "f"),
            "Input Cache (CNY)": format(values["input-cache"], "f"),
            "Output (CNY)": format(values["output"], "f"),
            "Total (CNY)": format(total, "f"),
        }
    )

  rows.sort(
      key=lambda row: (
          -ParseAmount(row["Total (CNY)"]),
          row["Customer"],
          row["Model"],
      )
  )
  return rows


def WriteCsv(output_csv: Path, rows: Sequence[Dict[str, str]]) -> None:
  """Writes the aggregated English CSV report."""
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", encoding="utf-8-sig", newline="") as target_file:
    writer = csv.DictWriter(target_file, fieldnames=CSV_HEADERS)
    writer.writeheader()
    writer.writerows(rows)


def FormatMoney(value: str, decimal_places: int = 6) -> str:
  """Formats a decimal amount with thousands separators."""
  return f"{ParseAmount(value):,.{decimal_places}f}"


def BuildTableRows(rows: Sequence[Dict[str, str]]) -> str:
  """Builds escaped HTML table rows."""
  html_rows = []
  for row in rows:
    html_rows.append(
        "\n".join(
            (
                "<tr>",
                f"  <td>{escape(row['Customer'])}</td>",
                f"  <td>{escape(row['Model'])}</td>",
                f"  <td class=\"number\">{FormatMoney(row['Input (CNY)'])}</td>",
                "  <td class=\"number\">"
                f"{FormatMoney(row['Input Cache (CNY)'])}</td>",
                f"  <td class=\"number\">{FormatMoney(row['Output (CNY)'])}</td>",
                "  <td class=\"number total\">"
                f"{FormatMoney(row['Total (CNY)'])}</td>",
                "</tr>",
            )
        )
    )
  return "\n".join(html_rows)


def BuildHtml(
    rows: Sequence[Dict[str, str]], title: str, top_count: int
) -> str:
  """Builds a standalone HTML report with a pie chart and data table."""
  grand_total = sum(
      (ParseAmount(row["Total (CNY)"]) for row in rows), Decimal("0")
  )
  table_headers = "".join(
      f"<th>{escape(header)}</th>" for header in CSV_HEADERS
  )
  table_rows = BuildTableRows(rows)
  safe_title = escape(title)

  template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; font-family: Inter, Arial, sans-serif; }
  body { margin: 0; padding: 28px; background: #f5f7fa; color: #1f2937; }
  .card { max-width: 1500px; margin: auto; background: white; border-radius: 12px;
          box-shadow: 0 4px 18px rgba(15, 23, 42, .08); overflow: hidden; }
  .summary { padding: 22px 24px; border-bottom: 1px solid #e5e7eb; }
  h1 { margin: 0 0 8px; font-size: 22px; }
  .meta { color: #64748b; font-size: 14px; }
  .chart-section { display: grid; grid-template-columns: minmax(320px, 520px) 1fr;
                   gap: 28px; align-items: center; padding: 24px;
                   border-bottom: 1px solid #e5e7eb; }
  .chart-wrap { position: relative; display: flex; justify-content: center; }
  #cost-chart { width: min(100%, 420px); height: auto; }
  .chart-legend { display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr));
                  gap: 10px 20px; }
  .legend-item { display: grid; grid-template-columns: 12px minmax(0, 1fr) auto;
                 gap: 8px; align-items: center; font-size: 12px; }
  .legend-color { width: 12px; height: 12px; border-radius: 3px; }
  .legend-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .legend-value { color: #475569; font-variant-numeric: tabular-nums; }
  .tooltip { position: fixed; display: none; pointer-events: none; z-index: 10;
             max-width: 360px; padding: 9px 11px; border-radius: 7px;
             background: rgba(15, 23, 42, .94); color: white; font-size: 12px;
             line-height: 1.5; box-shadow: 0 6px 20px rgba(15, 23, 42, .2); }
  .table-wrap { max-height: calc(100vh - 180px); overflow: auto; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
  th { position: sticky; top: 0; z-index: 1; padding: 12px 14px;
       background: #172554; color: white; text-align: left; white-space: nowrap; }
  td { padding: 10px 14px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }
  tbody tr:nth-child(even) { background: #f8fafc; }
  tbody tr:hover { background: #e0f2fe; }
  .number { text-align: right; font-variant-numeric: tabular-nums; }
  .total { font-weight: 700; color: #0f4c81; }
  @media (max-width: 900px) {
    .chart-section { grid-template-columns: 1fr; }
    .chart-legend { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="card">
  <div class="summary">
    <h1>__TITLE__</h1>
    <div class="meta">Sorted by Total (CNY), descending · __ROW_COUNT__ rows ·
      Total: ¥__GRAND_TOTAL__</div>
  </div>
  <div class="chart-section">
    <div class="chart-wrap">
      <canvas id="cost-chart" width="420" height="420"
              aria-label="Top cost distribution pie chart"></canvas>
    </div>
    <div id="chart-legend" class="chart-legend"></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>__TABLE_HEADERS__</tr></thead>
      <tbody>__TABLE_ROWS__</tbody>
    </table>
  </div>
</div>
<div id="chart-tooltip" class="tooltip"></div>
<script>
  (() => {
    const topCount = __TOP_COUNT__;
    const colors = [
      "#2563eb", "#7c3aed", "#db2777", "#e11d48", "#ea580c", "#ca8a04",
      "#16a34a", "#0891b2", "#4f46e5", "#9333ea", "#94a3b8"
    ];
    const sourceRows = Array.from(document.querySelectorAll("tbody tr")).map((row) => {
      const cells = row.querySelectorAll("td");
      return {
        name: `${cells[0].textContent} · ${cells[1].textContent}`,
        value: Number(cells[5].textContent.replaceAll(",", ""))
      };
    });
    const topRows = sourceRows.slice(0, topCount);
    const otherValue = sourceRows.slice(topCount).reduce((sum, row) => sum + row.value, 0);
    const chartData = otherValue > 0
      ? [...topRows, {name: "Others", value: otherValue}]
      : topRows;
    const total = chartData.reduce((sum, item) => sum + item.value, 0);
    const canvas = document.getElementById("cost-chart");
    const context = canvas.getContext("2d");
    const center = canvas.width / 2;
    const radius = center - 18;
    let angle = -Math.PI / 2;
    const slices = [];

    chartData.forEach((item, index) => {
      const start = angle;
      const end = total === 0 ? start : start + (item.value / total) * Math.PI * 2;
      context.beginPath();
      context.moveTo(center, center);
      context.arc(center, center, radius, start, end);
      context.closePath();
      context.fillStyle = colors[index % colors.length];
      context.fill();
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.stroke();
      slices.push({start, end, item});
      angle = end;
    });

    const legend = document.getElementById("chart-legend");
    chartData.forEach((item, index) => {
      const percentage = total === 0 ? 0 : (item.value / total) * 100;
      const entry = document.createElement("div");
      const color = document.createElement("span");
      const name = document.createElement("span");
      const value = document.createElement("span");
      entry.className = "legend-item";
      entry.title = item.name;
      color.className = "legend-color";
      color.style.background = colors[index % colors.length];
      name.className = "legend-name";
      name.textContent = item.name;
      value.className = "legend-value";
      value.textContent = `¥${item.value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      })} · ${percentage.toFixed(1)}%`;
      entry.append(color, name, value);
      legend.appendChild(entry);
    });

    const tooltip = document.getElementById("chart-tooltip");
    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (event.clientX - rect.left) * scaleX - center;
      const y = (event.clientY - rect.top) * scaleY - center;
      if (Math.hypot(x, y) > radius) {
        tooltip.style.display = "none";
        return;
      }
      let pointerAngle = Math.atan2(y, x);
      if (pointerAngle < -Math.PI / 2) pointerAngle += Math.PI * 2;
      const slice = slices.find(
        (candidate) => pointerAngle >= candidate.start && pointerAngle < candidate.end
      );
      if (!slice) return;
      const percentage = total === 0 ? 0 : (slice.item.value / total) * 100;
      tooltip.textContent =
        `${slice.item.name} · ¥${slice.item.value.toLocaleString("en-US", {
          minimumFractionDigits: 6,
          maximumFractionDigits: 6
        })} · ${percentage.toFixed(2)}%`;
      tooltip.style.display = "block";
      tooltip.style.left = `${event.clientX + 14}px`;
      tooltip.style.top = `${event.clientY + 14}px`;
    });
    canvas.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
  })();
</script>
</body>
</html>
"""
  replacements = {
      "__TITLE__": safe_title,
      "__ROW_COUNT__": str(len(rows)),
      "__GRAND_TOTAL__": f"{grand_total:,.6f}",
      "__TABLE_HEADERS__": table_headers,
      "__TABLE_ROWS__": table_rows,
      "__TOP_COUNT__": str(top_count),
  }
  for marker, value in replacements.items():
    template = template.replace(marker, value)
  return template


def WriteHtml(
    output_html: Path,
    rows: Sequence[Dict[str, str]],
    title: str,
    top_count: int,
) -> None:
  """Writes the standalone formatted HTML report."""
  output_html.parent.mkdir(parents=True, exist_ok=True)
  output_html.write_text(
      BuildHtml(rows, title, top_count), encoding="utf-8"
  )


def ResolveOutputPath(
    input_csv: Path, explicit_path: Path, suffix: str
) -> Path:
  """Returns an explicit output path or derives one from the input path."""
  if explicit_path:
    return explicit_path
  return input_csv.with_name(f"{input_csv.stem}-by-customer{suffix}")


def Main() -> None:
  """Runs CSV aggregation and report generation."""
  arguments = ParseArguments()
  if arguments.top < 1 or arguments.top > 10:
    raise ValueError("--top must be between 1 and 10")

  output_csv = ResolveOutputPath(
      arguments.input_csv, arguments.output_csv, ".csv"
  )
  output_html = ResolveOutputPath(
      arguments.input_csv, arguments.output_html, ".html"
  )
  rows = AggregateRows(arguments.input_csv)
  WriteCsv(output_csv, rows)
  WriteHtml(output_html, rows, arguments.title, arguments.top)
  print(f"CSV: {output_csv}")
  print(f"HTML: {output_html}")


if __name__ == "__main__":
  Main()

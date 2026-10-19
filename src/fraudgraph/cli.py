from .provenance import capture
import argparse
import json
from pathlib import Path
from .core import simulate, benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report, cases = benchmark(simulate())
    (out / "results.json").write_text(json.dumps(report, indent=2))
    (out / "investigations.json").write_text(json.dumps(cases, indent=2))
    text = "# FraudGraph investigation benchmark\n\nSynthetic data with delayed labels.\n\n"
    for name, r in report["models"].items():
        s = r["test"]
        text += f"- {name}: AP {s['average_precision']:.3f}, precision@capacity {s['precision_at_capacity']:.3f}, caught {s['caught']}, false positives {s['false_positives']}.\n"
    text += f"\nValidation-selected policy: {report['validation_selected']}. Shared-device evidence is an investigation lead, not proof of wrongdoing. Labels shown in case exports are retrospective evaluation outcomes.\n"
    (out / "summary.md").write_text(text)
    print(text)
    capture(out)


if __name__ == "__main__":
    main()

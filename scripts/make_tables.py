"""Write the README's result tables from results/*.json, or check that the README is up to date.

Every number in the README's results sections sits between <!-- table:NAME --> and <!-- /table:NAME -->
markers and is produced here, so a stale README fails CI instead of drifting from the data.

    python scripts/make_tables.py            # rewrite the tables in README.md
    python scripts/make_tables.py --check    # exit 1 if README.md differs from what the results give
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

GPU_TYPES = ("MISC", "T4", "P100", "V100", "V100M32")


def load(results: Path, name: str) -> dict:
    return json.loads((results / f"{name}.json").read_text())


def means(data: dict) -> dict[str, dict]:
    """Mean of every metric over seeds, per label."""
    out: dict[str, dict] = {}
    for run in data["runs"]:
        acc = out.setdefault(run["label"], {"n": 0})
        acc["n"] += 1
        for key, value in run["metrics"].items():
            if isinstance(value, dict):
                for sub, v in value.items():
                    acc[f"{key}.{sub}"] = acc.get(f"{key}.{sub}", 0.0) + v
            else:
                acc[key] = acc.get(key, 0.0) + value
    for acc in out.values():
        n = acc.pop("n")
        for key in acc:
            acc[key] /= n
    return out


def minutes(seconds: float) -> str:
    return f"{seconds / 60:,.1f}"


def reduction(c: dict) -> str:
    """'53% (95% CI 52 to 54%)': how much lower the candidate is, with the bootstrap interval over seeds."""
    return f"{-100 * c['mean']:.0f}% (95% CI {-100 * c['ci_high']:.0f} to {-100 * c['ci_low']:.0f}%)"


def change(c: dict) -> str:
    """'-54.8% [-56.0, -53.5]' with the bootstrap interval over seeds."""
    if c["mean"] is None:
        return f"{c['abs_mean'] / 60:+.1f} min [{c['abs_low'] / 60:+.1f}, {c['abs_high'] / 60:+.1f}]"
    return f"{100 * c['mean']:+.1f}% [{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]"


def delay_change(c: dict) -> str:
    """Absolute change in minutes: queueing delays are often near zero, where a percentage means little."""
    return f"{c['abs_mean'] / 60:+.1f} [{c['abs_low'] / 60:+.1f}, {c['abs_high'] / 60:+.1f}]"


def show(label: str, drop: str = "") -> str:
    """A run label as table text: 'aware-greedy | fleet 0.5' -> 'aware-greedy, fleet 50%' (a '|' would split
    the Markdown cell). `drop` removes a part the table's heading already states."""
    parts = [p for p in label.split(" | ") if p != drop]
    return ", ".join(f"fleet {float(p[6:]):.0%}" if p.startswith("fleet ") else p for p in parts)


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def comparison_rows(data: dict, wanted: list[tuple[str, str]] | None = None, drop: str = "") -> list[list[str]]:
    rows = []
    for c in data["comparisons"]:
        if wanted is not None and (c["baseline"], c["candidate"]) not in wanted:
            continue
        rows.append(
            [
                f"{show(c['candidate'], drop)} vs {show(c['baseline'], drop)}",
                change(c["mean_jct"]),
                change(c["p95_jct"]),
                delay_change(c["mean_wait"]),
                f"{c['mean_jct']['b_lower_in']} of {c['mean_jct']['seeds']}",
            ]
        )
    return rows


COMPARISON_HEADER = [
    "candidate vs baseline",
    "mean JCT",
    "p95 JCT",
    "mean queueing delay (min)",
    "seeds where mean JCT is lower",
]


def policy_rows(m: dict[str, dict], labels: list[str], drop: str = "") -> list[list[str]]:
    return [
        [
            show(label, drop),
            minutes(m[label]["mean_jct"]),
            minutes(m[label]["p95_jct"]),
            minutes(m[label]["mean_wait"]),
            f"{m[label]['makespan'] / 86400:.1f}",
            f"{m[label]['speed_fraction']:.2f}",
            f"{m[label]['jain_slowdown']:.2f}",
            f"{100 * m[label]['utilisation_all']:.0f}%",
            " / ".join(f"{100 * m[label][f'utilisation.{t}']:.0f}" for t in GPU_TYPES),
            f"{100 * m[label]['fragmentation_share']:.1f}%",
        ]
        for label in labels
        if label in m
    ]


POLICY_HEADER = [
    "policy",
    "mean JCT (min)",
    "p95 JCT (min)",
    "mean queueing delay (min)",
    "makespan (days)",
    "speed fraction",
    "Jain (slowdown)",
    "GPU use",
    "use per type, % (MISC / T4 / P100 / V100 / V100M32)",
    "waiting due to fragmentation",
]


def build(results: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    v = json.loads((results / "validation.json").read_text())
    out["validation"] = (
        table(
            ["case", "utilisation", "Wq theory (s)", "Wq simulated (s), 95% CI", "z", "L (time average)", "lambda x W"],
            [
                [
                    c["case"],
                    f"{c['rho']:.2f}",
                    f"{c['theory_wait']:.2f}",
                    f"{c['sim_wait']:.2f} +- {c['ci95']:.2f}",
                    f"{c['z']:+.2f}",
                    f"{c['L']:.3f}",
                    f"{c['lambda_W']:.3f}",
                ]
                for c in v["cases"]
            ],
        )
        + f"\n\n{v['cases'][0]['reps']} replications of {v['cases'][0]['jobs_per_rep']:,} jobs per case, "
        f"first 10% of each discarded; mean service time {v['mean_service']:.0f} s."
    )

    main = load(results, "main")
    head = main["comparisons"][0]
    assert (head["baseline"], head["candidate"]) == ("blind-backfill", "aware-greedy")
    m = means(main)
    rank = next(c for c in main["comparisons"] if c["candidate"] == "rank-greedy")
    extra = next(c for c in main["comparisons"] if (c["baseline"], c["candidate"]) == ("rank-greedy", "aware-greedy"))
    busy_base = "blind-backfill | fleet 0.3"
    busy = {c["candidate"]: c for c in load(results, "load")["comparisons"] if c["baseline"] == busy_base}
    busy_wait = means(load(results, "load"))[busy_base]["mean_wait"]
    out["headline"] = "\n".join(
        [
            "- **On the trace's own fleet and load, knowing the GPU type pays a lot.** A scheduler that sends each "
            "job to its fastest free GPU type (`aware-greedy`) cuts the mean job completion time by "
            f"**{reduction(head['mean_jct'])}** against a type-blind scheduler (`blind-backfill`): from "
            f"{minutes(m['blind-backfill']['mean_jct'])} to {minutes(m['aware-greedy']['mean_jct'])} minutes, lower in "
            f"{head['mean_jct']['b_lower_in']} of {head['mean_jct']['seeds']} paired seeds. The p95 completion time "
            f"falls by {reduction(head['p95_jct'])}.",
            f"- **Most of it needs only a ranking of the types.** Always trying the type that is fastest on average "
            f"first (`rank-greedy`) gives {reduction(rank['mean_jct'])}. Knowing each job's own speeds cuts another "
            f"{reduction(extra['mean_jct'])} off `rank-greedy`.",
            f"- **On a busy fleet, placing greedily is not enough.** With 30% of the GPUs, where jobs wait "
            f"{minutes(busy_wait)} minutes on average under `blind-backfill`, `aware-greedy` gains only "
            f"{reduction(busy['aware-greedy | fleet 0.3']['mean_jct'])}. Waiting for a busy fast GPU when that "
            f"finishes the job sooner (`aware-wait`) gains {reduction(busy['aware-wait | fleet 0.3']['mean_jct'])}.",
        ]
    )
    labels = ["blind-fifo", "blind-backfill", "rank-greedy", "aware-greedy", "aware-wait", "aware-lp"]
    out["main"] = (
        table(POLICY_HEADER, policy_rows(m, labels)) + "\n\n" + table(COMPARISON_HEADER, comparison_rows(main))
    )

    load_data = load(results, "load")
    lm = means(load_data)
    parts = []
    for fleet in ("0.5", "0.3"):
        fleet_labels = [f"{p} | fleet {fleet}" for p in labels]
        drop = f"fleet {fleet}"
        parts.append(
            f"**Fleet at {float(fleet):.0%} of its GPUs**\n\n"
            + table(POLICY_HEADER, policy_rows(lm, fleet_labels, drop))
        )
        wanted = [(c["baseline"], c["candidate"]) for c in load_data["comparisons"] if c["baseline"].endswith(drop)]
        parts.append(table(COMPARISON_HEADER, comparison_rows(load_data, wanted, drop)))
    out["load"] = "\n\n".join(parts)

    mig = load(results, "migration")
    mm = means(mig)
    rows = []
    for c in mig["comparisons"]:
        cand = c["candidate"]
        rows.append(
            [
                f"{show(cand)} vs {show(c['baseline'])}",
                minutes(mm[cand]["mean_jct"]),
                change(c["mean_jct"]),
                change(c["p95_jct"]),
                f"{mm[cand]['migrations']:,.0f}",
                f"{100 * mm[cand]['migrated_share']:.1f}%",
            ]
        )
    out["migration"] = table(
        [
            "candidate vs baseline",
            "mean JCT (min)",
            "mean JCT change",
            "p95 JCT change",
            "moves per seed",
            "jobs moved",
        ],
        rows,
    )

    sh = load(results, "sharing")
    shm = means(sh)
    rows = []
    for c in sh["comparisons"]:
        a, b = shm[c["baseline"]], shm[c["candidate"]]
        rows.append(
            [
                show(c["candidate"]),
                f"{minutes(a['mean_jct'])} -> {minutes(b['mean_jct'])}",
                change(c["mean_jct"]),
                delay_change(c["mean_wait"]),
                f"{100 * a['utilisation_all']:.0f}% -> {100 * b['utilisation_all']:.0f}%",
                f"{100 * a['fragmentation_share']:.1f}% -> {100 * b['fragmentation_share']:.1f}%",
            ]
        )
    out["sharing"] = table(
        [
            "policy",
            "mean JCT (min), off -> on",
            "mean JCT change",
            "mean queueing delay change (min)",
            "GPU capacity in use, off -> on",
            "waiting due to fragmentation, off -> on",
        ],
        rows,
    )

    sens = load(results, "sensitivity")
    sm = means(sens)
    comps = {(c["baseline"], c["candidate"]): c for c in sens["comparisons"]}
    rows = []
    for gamma in ("0.0", "0.5", "1.0", "1.5", "2.0", "uniform"):

        def lab(p: str, gamma: str = gamma) -> str:
            if gamma == "uniform":
                return f"{p} | uniform speeds"
            return p if gamma == "1.0" else f"{p} | gamma {gamma}"

        blind, rk, aw = lab("blind-backfill"), lab("rank-greedy"), lab("aware-greedy")
        rows.append(
            [
                "same speed row for every job (gamma 1)" if gamma == "uniform" else gamma,
                minutes(sm[blind]["mean_jct"]),
                minutes(sm[rk]["mean_jct"]),
                minutes(sm[aw]["mean_jct"]),
                change(comps[(blind, rk)]["mean_jct"]),
                change(comps[(blind, aw)]["mean_jct"]),
            ]
        )
    out["sensitivity"] = table(
        [
            "gamma",
            "blind-backfill (min)",
            "rank-greedy (min)",
            "aware-greedy (min)",
            "rank-greedy vs blind",
            "aware-greedy vs blind",
        ],
        rows,
    )

    est = load(results, "estimates")
    em = means(est)
    comps = {(c["baseline"], c["candidate"]): c for c in est["comparisons"]}
    rows = [["0 (exact)", minutes(em["aware-greedy"]["mean_jct"]), "", "", ""]]
    for noise in ("0.25", "0.5", "1.0"):
        cand = f"aware-greedy~noise{noise}"
        rows.append(
            [
                noise,
                minutes(em[cand]["mean_jct"]),
                change(comps[("blind-backfill", cand)]["mean_jct"]),
                change(comps[("rank-greedy", cand)]["mean_jct"]),
                change(comps[("aware-greedy", cand)]["mean_jct"]),
            ]
        )
    out["estimates"] = table(
        ["error (sd of log speed)", "mean JCT (min)", "vs blind-backfill", "vs rank-greedy", "vs exact speeds"], rows
    )
    return out


def render(readme: str, tables: dict[str, str]) -> str:
    for name, body in tables.items():
        # the markers may be empty (a fresh README) or hold an earlier version of the table
        pattern = re.compile(rf"(<!-- table:{name} -->\n)(?:.*?\n)?(<!-- /table:{name} -->)", re.S)
        if not pattern.search(readme):
            raise SystemExit(f"README has no markers for table '{name}'")
        readme = pattern.sub(lambda match, body=body: match.group(1) + body + "\n" + match.group(2), readme)
    return readme


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-structure", action="store_true", help="only check that every table can be built")
    args = parser.parse_args()
    current = args.readme.read_text(encoding="utf-8")
    updated = render(current, build(args.results))
    if args.check_structure:
        print(f"all tables build from {args.results}")
        return 0
    if args.check:
        if updated != current:
            print("README.md tables are out of date: run python scripts/make_tables.py", file=sys.stderr)
            return 1
        print("README.md tables match results/")
        return 0
    args.readme.write_text(updated, encoding="utf-8", newline="\n")
    print("README.md tables updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

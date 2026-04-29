"""
For each total number of matrix multiplications, enumerate all multisets of
polynomial degrees (q values from degree_mults) whose per-step multiplication
counts sum to that total.  For every such combination, run the framework's
get_all_coeffs_different_degrees and then propagate l_init through each fitted
polynomial to obtain the final l value.  A larger final l means the composition
has converged closer to sign(x)=1, i.e. less error.

Usage:
    python best_combinations.py
"""

import contextlib
import io
import os
import sys
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np

# Allow importing from framework/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "framework"))
from framework import get_all_coeffs_different_degrees  # noqa: E402

# Maps polynomial degree parameter q -> number of matrix multiplications needed
degree_mults = {2: 3, 3: 5, 4: 9, 5: 17, 6: 25, 7: 41, 8: 61}


def p(c, x):
    """Evaluate odd polynomial p(x) = sum c[i] * x^(2i+1)."""
    out = 0
    x_power = x  # x^(2*0+1) = x
    x2 = x * x
    for ci in c:
        out += ci * x_power
        x_power *= x2
    return out


def combinations_by_total_mults(target_mults):
    """
    Return all multisets of q values (keys of degree_mults) such that
    sum(degree_mults[q_i] for q_i in combo) == target_mults.
    Uses combinations_with_replacement so each multiset is represented once
    in sorted order (consistent with combinations.py).
    """
    q_values = sorted(degree_mults.keys())
    min_mults = min(degree_mults.values())
    results = []
    max_len = target_mults // min_mults
    for r in range(1, max_len + 1):
        for comb in combinations_with_replacement(q_values, r):
            if sum(degree_mults[q] for q in comb) == target_mults:
                results.append(list(comb))
    return results


def compute_final_l(q_list, l_init=0.001):
    """
    Run get_all_coeffs_different_degrees with q_list, then propagate l_init
    through each fitted polynomial to obtain the final l value.

    Returns (final_l, coeffs) on success, or (None, None) on failure.
    A larger final_l means less approximation error at the end.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coeffs = get_all_coeffs_different_degrees(
                list(q_list), len(q_list), l_init
            )
        l = l_init
        for c in coeffs:
            l = float(p(c, l))
        return l, coeffs
    except Exception:
        return None, None


def find_best_combinations(max_target_mults=30, l_init=0.001):
    """
    For each achievable total number of multiplications up to max_target_mults,
    compute the final l for every valid degree combination and record the best
    (largest l) one.

    Returns a dict: {target_mults: {'best_combo', 'best_l', 'all_combos'}}.
    """
    results = {}
    min_mults = min(degree_mults.values())

    for target_mults in range(min_mults, max_target_mults + 1):
        combos = combinations_by_total_mults(target_mults)
        if not combos:
            continue

        best_l = -np.inf
        best_combo = None
        combo_results = []

        for combo in combos:
            final_l, _ = compute_final_l(combo, l_init)
            if final_l is not None:
                combo_results.append((combo, final_l))
                if final_l > best_l:
                    best_l = final_l
                    best_combo = combo

        if best_combo is not None:
            results[target_mults] = {
                "best_combo": best_combo,
                "best_l": best_l,
                "all_combos": combo_results,
            }
            mults_per_step = [degree_mults[q] for q in best_combo]
            print(
                f"Total mults: {target_mults:3d} | "
                f"Best q combo: {best_combo} | "
                f"Mults/step: {mults_per_step} | "
                f"Final l: {best_l:.8f}"
            )

    return results


def plot_best_l(results):
    """Plot final l and the best combination label for each total mult count."""
    mults = sorted(results.keys())
    best_ls = [results[m]["best_l"] for m in mults]
    labels = [str(results[m]["best_combo"]) for m in mults]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mults, best_ls, "o-", linewidth=1.5)
    axes[0].set_xlabel("Total multiplications")
    axes[0].set_ylabel("Final l  (larger = less error)")
    axes[0].set_title("Best final l per total multiplications")
    axes[0].grid(True)

    axes[1].bar(range(len(mults)), best_ls)
    axes[1].set_xticks(range(len(mults)))
    axes[1].set_xticklabels(
        [f"{m}\n{labels[i]}" for i, m in enumerate(mults)],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axes[1].set_xlabel("Total mults  (best q combo below)")
    axes[1].set_ylabel("Final l")
    axes[1].set_title("Best combination per total multiplications")

    plt.tight_layout()
    plt.show()


def main():
    print("Searching for best degree combinations by total multiplications...")
    print("(Best = largest final l value, i.e. least approximation error)\n")

    results = find_best_combinations(max_target_mults=30)

    if not results:
        print("No valid combinations found.")
        return

    print("\n--- Summary (sorted by total mults) ---")
    for m in sorted(results.keys()):
        r = results[m]
        mults_per_step = [degree_mults[q] for q in r["best_combo"]]
        print(
            f"  Total mults {m:3d}: best q = {r['best_combo']}, "
            f"mults/step = {mults_per_step}, "
            f"final l = {r['best_l']:.8f}"
        )

    plot_best_l(results)


if __name__ == "__main__":
    main()

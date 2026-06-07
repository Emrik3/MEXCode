from itertools import combinations_with_replacement

import matplotlib.pyplot as plt

degree_mults = {2: 3, 3: 5, 4: 9, 5: 17}  # , 6: 25, 7: 41, 8: 61
dof = {2: 2, 3: 3, 4: 5, 5: 9}  # , 6: 16, 7: 25, 8: 36


def combinations(total_degree):
    degrees = sorted(degree_mults.keys())
    results = []
    max_len = total_degree // min(degrees)
    for r in range(1, max_len + 1):
        for comb in combinations_with_replacement(degrees, r):
            if sum(comb) == total_degree:
                results.append(list(comb))
    return results


def achieved_degree(combo):
    prod = 1
    for d in combo:
        prod *= (degree_mults[d] - 1) / 2 + 1
    return prod


def achieved_dof(combo):
    return sum(dof[d] for d in combo)


def bar_plot_degree(target_degree):

    combos = combinations(target_degree)

    # (combo, achieved degree, dof)
    combo_data = [(c, achieved_degree(c), achieved_dof(c)) for c in combos]

    # Sort by achieved degree
    combo_data.sort(key=lambda x: x[1])

    labels = [str(c) for c, _, _ in combo_data]
    degree_values = [deg for _, deg, _ in combo_data]
    dof_values = [d for _, _, d in combo_data]
    total_mults = [sum(c) for c, _, _ in combo_data]

    # Color map by total degree sum (same logic as DoF plot)
    unique_totals = sorted(set(total_mults))
    cmap = plt.cm.get_cmap("tab20")
    color_map = {t: cmap(i % 20) for i, t in enumerate(unique_totals)}
    colors = [color_map[t] for t in total_mults]

    plt.figure()
    bars = plt.bar(range(len(degree_values)), degree_values, color=colors)

    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("q+1 (Product)")
    plt.xlabel("Combinations of Degrees")
    plt.title(f"Achieved Degree vs Combinations (sum ≤ {target_degree})")

    # Annotate DoF above each bar
    for bar, d in zip(bars, dof_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{d}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    # Legend (same semantic meaning as DoF plot)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color_map[t], label=f"Total sum: {t}") for t in unique_totals
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.show()


def max_comb(max_degree):
    for target_degree in range(2, max_degree):
        combos = combinations(target_degree)

        # (combo, achieved degree, dof)
        combo_data = [(c, achieved_degree(c), achieved_dof(c)) for c in combos]

        combo_data.sort(key=lambda x: x[1])

        best = combo_data[-1][0]
        best_l = [0.0 for i in range(len(best))]
        for i in range(len(best)):
            best[i] = degree_mults[best[i]]
        power = 1
        for q in best:
            power *= (q - 1) / 2 + 1
        best_l = 1 - (1 - 0.001) ** power

        print(
            f"Total multiplications: {target_degree} has the best combination of degrees {best} with l = {best_l}"
        )


def main():
    max_comb(26)


if __name__ == "__main__":
    main()

from itertools import combinations_with_replacement
import matplotlib.pyplot as plt

# degree_mults = {2: 3, 3: 5, 4: 9, 5: 17, 6: 31} # Degree 5 and 6 mults from sastre
# dof = {2: 2, 3: 3, 4: 5, 5: 9, 6: 16}

degree_mults = {2: 3, 3: 5, 4: 9, 5: 17}
dof = {2: 2, 3: 3, 4: 5, 5: 9}

def bar_plot_dof(target_degree):

    def combinations(total_degree):
        degrees = sorted(degree_mults.keys())
        results = []
        max_len = total_degree // min(degrees)
        for r in range(1, max_len + 1):
            for comb in combinations_with_replacement(degrees, r):
                if sum(comb) <= total_degree:
                    results.append(list(comb))
        return results

    def achieved_degree(combo):
        prod = 1
        for d in combo:
            prod *= degree_mults[d]
        return prod

    def achieved_dof(combo):
        return sum(dof[d] for d in combo)

    combos = combinations(target_degree)

    # (combo, total dof, achieved order)
    combo_data = [
        (c, achieved_dof(c), achieved_degree(c)) for c in combos
    ]

    # Sort by degrees of freedom
    combo_data.sort(key=lambda x: x[1])

    labels = [str(c) for c, _, _ in combo_data]
    dof_values = [d for _, d, _ in combo_data]
    degree_values = [deg for _, _, deg in combo_data]
    total_mults = [sum(c) for c, _, _ in combo_data]

    # Create color map based on total number of multiplications
    unique_totals = sorted(set(total_mults))
    cmap = plt.cm.get_cmap('tab20')  # More distinct colors
    # Map each unique total to an index for better color separation
    color_map = {t: cmap(i % 20) for i, t in enumerate(unique_totals)}
    colors = [color_map[t] for t in total_mults]

    plt.figure()
    bars = plt.bar(range(len(dof_values)), dof_values, color=colors)

    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Degrees of Freedom")
    plt.xlabel("Combinations of Degrees")
    plt.title(f"Degrees of Freedom vs Multiplication Combinations (sum = {target_degree})")

    # Annotate achieved total degree above each bar
    for bar, deg in zip(bars, degree_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(deg),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45
        )

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[t], label=f'Total mults: {t}') 
                       for t in unique_totals]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()

def bar_plot_degree(target_degree):

    def combinations(total_degree):
        degrees = sorted(degree_mults.keys())
        results = []
        max_len = total_degree // min(degrees)
        for r in range(1, max_len + 1):
            for comb in combinations_with_replacement(degrees, r):
                if sum(comb) <= total_degree:
                    results.append(list(comb))
        return results

    def achieved_degree(combo):
        prod = 1
        for d in combo:
            prod *= degree_mults[d] + 1
        return prod

    def achieved_dof(combo):
        return sum(dof[d] for d in combo)

    combos = combinations(target_degree)

    # (combo, achieved degree, dof)
    combo_data = [
        (c, achieved_degree(c), achieved_dof(c)) for c in combos
    ]

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
    plt.ylabel("Achieved Degree (Product)")
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
            rotation=45
        )

    # Legend (same semantic meaning as DoF plot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[t], label=f"Total sum: {t}")
        for t in unique_totals
    ]
    plt.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.show()


def main():
    bar_plot_degree(15)


if __name__ == "__main__":
    main()

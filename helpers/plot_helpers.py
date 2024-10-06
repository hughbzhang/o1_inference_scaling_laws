import matplotlib.pyplot as plt
import json
import typing
import logging

GRAPH_FOLDER = "graphs"
HELPER_FOLDER = "helpers"

def plot_majority_vote_graph(results: list[dict[str, typing.Any]], shade_regions: bool = False) -> None:
    """
    Plot the majority vote graph.

    Args:
        results (list[dict[str, typing.Any]]): The results to plot.
    """
    plt.figure(figsize=(5, 6))
    plt.scatter([r['avg_tokens_used'] for r in results], [100*r['accuracy'] for r in results], marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('tokens used at test-time (log scale)', fontsize=13)
    plt.ylabel('pass@1 accuracy', fontsize=13)
    plt.ylim(0, 100)
    plt.title('o1 mini AIME accuracy\nat test time (reconstructed)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    if shade_regions:
        plt.axvline(x=2**14, color='black', linestyle='--')

        plt.axvspan(min([r['avg_tokens_used'] for r in results]) // 2, 2**14, facecolor='lightgreen', alpha=0.3)
        plt.axvspan(2**14, 2**17, facecolor='lightblue', alpha=0.3)

        plt.text(2**11, 85, "just ask o1-mini \nto think longer", fontsize=12, ha='center', va='center', color='green')
        plt.text(2**15*1.4, 85, 'majority\nvote', fontsize=12, ha='center', va='center', color='blue')

        plt.axvline(x=2**17, color='black', linestyle='--')
        plt.text(2**19, 85, 'no gains', fontsize=12, ha='center', va='center', color='red')
        plt.axvspan(2**17, 2**21, facecolor='lightcoral', alpha=0.3)


    plt.tight_layout()
    accuracy_vs_tokens_plot_name = 'accuracy_vs_tokens_{}.png'.format("shade_regions" if shade_regions else "no_shade_regions")
    plt.savefig(f"{GRAPH_FOLDER}/{accuracy_vs_tokens_plot_name}", dpi=300, facecolor='white', edgecolor='none')
    plt.close()

    print(f"Plot saved as {accuracy_vs_tokens_plot_name}")

    if not shade_regions:
        plt.figure(figsize=(11, 10))
        plt.scatter([r['token_limit'] for r in results], [r['avg_tokens_used'] for r in results], marker='o')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.xlabel('Token Limit')
        plt.ylabel('Actual Tokens Used')
        plt.title('Token Limit vs. Actual Tokens Used')
        plt.tight_layout()
        token_limit_vs_actual_plot_name = 'token_limit_vs_actual.png'
        plt.savefig(f"{GRAPH_FOLDER}/{token_limit_vs_actual_plot_name}")

        with open(f'{HELPER_FOLDER}/results_log_majority_vote.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"plots saved to {token_limit_vs_actual_plot_name}")

def plot_just_ask_nicely_graph(results: list[dict[str, typing.Any]], run_full_range: bool = False) -> None:
    """
    Plot the just ask nicely graph.

    Args:
        results (list[dict[str, typing.Any]]): The results to plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter([r['token_limit'] for r in results], [r['avg_tokens_used'] for r in results], marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Token Limit')
    plt.ylabel('Actual Tokens Used')
    plt.title('Token Limit vs. Actual Tokens Used')
    plt.tight_layout()
    if run_full_range:
        plt_name = f"{GRAPH_FOLDER}/full_just_ask_nicely.png"
    else:
        plt_name = f"{GRAPH_FOLDER}/just_ask_nicely_tokens.png"

        with open(f'{HELPER_FOLDER}/results_log_just_ask_nicely.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    plt.savefig(plt_name)
    print(f"plots saved to {plt_name}")
import matplotlib.pyplot as plt

def plot_convergence(histories, title="Convergence Comparison"):
    plt.figure(figsize=(10, 6))
    for name, losses in histories.items():
        plt.plot(losses, label=name)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.optimizers_numpy import GradientDescent, Momentum, Nesterov, Adam, RMSProp
from src.test_functions import rosenbrock, grad_rosenbrock
from utils.visualization import plot_convergence

def run():
    x0 = np.array([-1.0, 1.0])
    opts = {
        'GD': GradientDescent(0.001),
        'Momentum': Momentum(0.001),
        'Nesterov': Nesterov(0.001),
        'Adam': Adam(0.01)
    }
    histories = {}
    print("Running Math Function Experiments...")
    for name, opt in opts.items():
        x = x0.copy()
        losses = []
        for _ in range(1000):
            losses.append(rosenbrock(x))
            if isinstance(opt, Nesterov): x = opt.step(x, grad_rosenbrock)
            else: x = opt.step(x, grad_rosenbrock(x))
        histories[name] = losses
    
    plot_convergence(histories, "Rosenbrock Convergence")
    print("Experiment Done. Plot saved.")

if __name__ == "__main__": run()


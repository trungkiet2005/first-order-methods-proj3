import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import MLP, SimpleCNN
from src.optimizers_torch import get_optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def evaluate(model, loader, criterion, device):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_model(model, optimizer_name, train_loader, test_loader, device, epochs=5, lr=0.001):
    """Train a model with specified optimizer"""
    print(f"\n{'='*60}")
    print(f"Training {model.__class__.__name__} with {optimizer_name.upper()}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")
    
    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Average Epoch Time: {sum(epoch_times)/len(epoch_times):.2f}s")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_acc': test_accs[-1]
    }

def plot_results(results_dict, model_name):
    """Plot training results for all optimizers"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} Training Results - Optimizer Comparison', fontsize=16, fontweight='bold')
    
    # Plot training loss
    ax = axes[0, 0]
    for opt_name, results in results_dict.items():
        ax.plot(results['train_losses'], label=opt_name, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot test loss
    ax = axes[0, 1]
    for opt_name, results in results_dict.items():
        ax.plot(results['test_losses'], label=opt_name, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot training accuracy
    ax = axes[1, 0]
    for opt_name, results in results_dict.items():
        ax.plot(results['train_accs'], label=opt_name, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax = axes[1, 1]
    for opt_name, results in results_dict.items():
        ax.plot(results['test_accs'], label=opt_name, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {model_name}_optimizer_comparison.png")

def plot_final_comparison(mlp_results, cnn_results):
    """Plot final accuracy comparison between MLP and CNN"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract final accuracies
    optimizers = list(mlp_results.keys())
    mlp_accs = [mlp_results[opt]['final_acc'] for opt in optimizers]
    cnn_accs = [cnn_results[opt]['final_acc'] for opt in optimizers]
    
    # Bar chart comparison
    x = range(len(optimizers))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], mlp_accs, width, label='MLP', alpha=0.8)
    ax1.bar([i + width/2 for i in x], cnn_accs, width, label='CNN', alpha=0.8)
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Final Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([opt.upper() for opt in optimizers])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Table of results
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = [['Optimizer', 'MLP Acc (%)', 'CNN Acc (%)', 'Difference']]
    for opt in optimizers:
        mlp_acc = mlp_results[opt]['final_acc']
        cnn_acc = cnn_results[opt]['final_acc']
        diff = cnn_acc - mlp_acc
        table_data.append([
            opt.upper(),
            f"{mlp_acc:.2f}",
            f"{cnn_acc:.2f}",
            f"{diff:+.2f}"
        ])
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('MLP_vs_CNN_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: MLP_vs_CNN_comparison.png")

def main():
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Data loading
    print("\n📥 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Optimizers to compare
    optimizers_to_test = ['sgd', 'momentum', 'adam', 'rmsprop']
    learning_rates = {
        'sgd': 0.01,
        'momentum': 0.01,
        'adam': 0.001,
        'rmsprop': 0.001
    }
    
    epochs = 5
    
    # Train MLP with different optimizers
    print("\n" + "="*60)
    print("TRAINING MLP (Multi-Layer Perceptron)")
    print("="*60)
    mlp_results = {}
    
    for opt_name in optimizers_to_test:
        model = MLP().to(device)
        lr = learning_rates[opt_name]
        results = train_model(model, opt_name, train_loader, test_loader, device, epochs, lr)
        mlp_results[opt_name] = results
    
    # Plot MLP results
    plot_results(mlp_results, 'MLP')
    
    # Train CNN with different optimizers
    print("\n" + "="*60)
    print("TRAINING CNN (Convolutional Neural Network)")
    print("="*60)
    cnn_results = {}
    
    for opt_name in optimizers_to_test:
        model = SimpleCNN().to(device)
        lr = learning_rates[opt_name]
        results = train_model(model, opt_name, train_loader, test_loader, device, epochs, lr)
        cnn_results[opt_name] = results
    
    # Plot CNN results
    plot_results(cnn_results, 'CNN')
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\n📊 MLP Results:")
    for opt_name in optimizers_to_test:
        print(f"  {opt_name.upper():10s}: {mlp_results[opt_name]['final_acc']:.2f}%")
    
    print("\n📊 CNN Results:")
    for opt_name in optimizers_to_test:
        print(f"  {opt_name.upper():10s}: {cnn_results[opt_name]['final_acc']:.2f}%")
    
    # Plot final comparison
    plot_final_comparison(mlp_results, cnn_results)
    
    print("\n" + "="*60)
    print("✅ All experiments completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

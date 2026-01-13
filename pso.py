import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from glob import glob
import warnings
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION (Same as GNN code)
# ============================================================================

def calculate_rms_error(healthy_signal, damaged_signal):
    """Calculate RMS error between healthy and damaged signals"""
    return np.sqrt(np.mean((healthy_signal - damaged_signal) ** 2))

def extract_dominant_frequency(signal_data, sampling_rate=100):
    """Extract dominant frequency from signal using FFT"""
    n = len(signal_data)
    yf = fft(signal_data)
    xf = fftfreq(n, 1/sampling_rate)
    
    positive_freq_idx = xf > 0
    xf_pos = xf[positive_freq_idx]
    yf_pos = np.abs(yf[positive_freq_idx])
    
    dominant_idx = np.argmax(yf_pos)
    return xf_pos[dominant_idx]

def calculate_frequency_shift(healthy_signal, damaged_signal, sampling_rate=100):
    """Calculate frequency shift between healthy and damaged signals"""
    healthy_freq = extract_dominant_frequency(healthy_signal, sampling_rate)
    damaged_freq = extract_dominant_frequency(damaged_signal, sampling_rate)
    return abs(damaged_freq - healthy_freq)

# ============================================================================
# GPU-ACCELERATED PSO IMPLEMENTATION
# ============================================================================

class BridgePSO:
    """GPU-accelerated Particle Swarm Optimization for damage localization"""
    
    def __init__(self, n_particles=50, n_elements=3, bounds=(0.1, 1.0), 
                 c1=0.5, c2=0.3, w=0.9, device='cuda'):
        """
        Initialize PSO optimizer
        
        Args:
            n_particles: Number of particles in swarm
            n_elements: Number of structural elements (sensors)
            bounds: Tuple of (lower, upper) bounds for stiffness reduction
            c1: Cognitive parameter
            c2: Social parameter
            w: Inertia weight
            device: 'cuda' or 'cpu'
        """
        self.n_particles = n_particles
        self.n_elements = n_elements
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # PSO parameters
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bounds = bounds
        
        # Initialize particles (stiffness reduction factors for each element)
        # 1.0 = healthy, < 1.0 = damaged
        self.positions = torch.rand(n_particles, n_elements, device=self.device)
        self.positions = self.positions * (bounds[1] - bounds[0]) + bounds[0]
        
        # Initialize velocities
        self.velocities = torch.randn(n_particles, n_elements, device=self.device) * 0.1
        
        # Best positions
        self.personal_best_positions = self.positions.clone()
        self.personal_best_scores = torch.full((n_particles,), float('inf'), device=self.device)
        
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        # History tracking
        self.history = {
            'global_best_scores': [],
            'mean_scores': [],
            'std_scores': []
        }
    
    def apply_damage_to_signals(self, healthy_signals, damage_factors):
        """
        Apply damage factors to healthy signals to simulate damaged response
        
        Args:
            healthy_signals: Tensor of shape (n_sensors, signal_length)
            damage_factors: Tensor of shape (n_sensors,) with values in [0, 1]
        
        Returns:
            Simulated damaged signals
        """
        # Simple damage model: reduce amplitude and shift frequency
        # This is a simplified physical model
        damaged = healthy_signals.clone()
        
        for i, factor in enumerate(damage_factors):
            # Amplitude reduction
            damaged[i] *= factor
            
            # Add noise proportional to damage severity
            noise_level = (1 - factor) * 0.1
            damaged[i] += torch.randn_like(damaged[i]) * noise_level * damaged[i].std()
        
        return damaged
    
    def fitness_function(self, particles, healthy_signals, observed_signals):
        """
        Evaluate fitness for all particles (GPU-accelerated)
        
        Args:
            particles: Tensor of shape (n_particles, n_elements)
            healthy_signals: Tensor of shape (n_sensors, signal_length)
            observed_signals: Tensor of shape (n_sensors, signal_length)
        
        Returns:
            Fitness scores for each particle (lower is better)
        """
        batch_size = particles.shape[0]
        n_sensors = healthy_signals.shape[0]
        
        fitness_scores = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # Simulate damaged response with current particle's damage factors
            simulated = self.apply_damage_to_signals(healthy_signals, particles[i])
            
            # Calculate RMS error between simulated and observed
            error = torch.sqrt(torch.mean((simulated - observed_signals) ** 2))
            
            # Add regularization to prevent trivial solutions (all damaged)
            # Encourage sparsity in damage
            sparsity_penalty = torch.sum(1.0 - particles[i]) * 0.01
            
            fitness_scores[i] = error + sparsity_penalty
        
        return fitness_scores
    
    def update(self, fitness_scores):
        """Update particle positions and velocities"""
        # Update personal bests
        improved = fitness_scores < self.personal_best_scores
        self.personal_best_scores[improved] = fitness_scores[improved]
        self.personal_best_positions[improved] = self.positions[improved].clone()
        
        # Update global best
        min_idx = torch.argmin(fitness_scores)
        if fitness_scores[min_idx] < self.global_best_score:
            self.global_best_score = fitness_scores[min_idx].item()
            self.global_best_position = self.positions[min_idx].clone()
        
        # Update velocities
        r1 = torch.rand_like(self.positions)
        r2 = torch.rand_like(self.positions)
        
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)
        
        self.velocities = self.w * self.velocities + cognitive + social
        
        # Update positions
        self.positions = self.positions + self.velocities
        
        # Apply bounds
        self.positions = torch.clamp(self.positions, self.bounds[0], self.bounds[1])
    
    def optimize(self, healthy_signals, observed_signals, n_iterations=50, verbose=True):
        """
        Run PSO optimization
        
        Args:
            healthy_signals: Tensor of healthy sensor signals
            observed_signals: Tensor of observed (damaged) sensor signals
            n_iterations: Number of PSO iterations
            verbose: Print progress
        
        Returns:
            Best damage factors found
        """
        iterator = tqdm(range(n_iterations)) if verbose else range(n_iterations)
        
        for iteration in iterator:
            # Evaluate fitness
            fitness_scores = self.fitness_function(
                self.positions, 
                healthy_signals, 
                observed_signals
            )
            
            # Update particles
            self.update(fitness_scores)
            
            # Track history
            self.history['global_best_scores'].append(self.global_best_score)
            self.history['mean_scores'].append(fitness_scores.mean().item())
            self.history['std_scores'].append(fitness_scores.std().item())
            
            if verbose and iteration % 10 == 0:
                iterator.set_description(
                    f"Best Fitness: {self.global_best_score:.6f}"
                )
        
        return self.global_best_position

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_sensor_data(csv_path, sensor_names=['Sensor_24', 'Sensor_31', 'Sensor_32']):
    """Load sensor data from CSV file"""
    df = pd.read_csv(csv_path)
    
    signals = []
    for sensor in sensor_names:
        # Get all columns for this sensor
        sensor_cols = [col for col in df.columns if sensor in col]
        if len(sensor_cols) > 0:
            # Take first acceleration column
            signal = df[sensor_cols[0]].values
            signals.append(signal)
    
    return np.array(signals)

def prepare_signals_for_pso(signals, max_length=1000):
    """Convert numpy signals to PyTorch tensors"""
    # Truncate or pad to consistent length
    processed = []
    for signal in signals:
        if len(signal) > max_length:
            signal = signal[:max_length]
        elif len(signal) < max_length:
            signal = np.pad(signal, (0, max_length - len(signal)), mode='constant')
        processed.append(signal)
    
    return torch.tensor(np.array(processed), dtype=torch.float32)

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_pso_results(predicted_damage, true_damage, threshold=0.9):
    """
    Evaluate PSO damage localization results
    
    Args:
        predicted_damage: Array of predicted stiffness factors
        true_damage: Array of true stiffness factors (ground truth)
        threshold: Values above this are considered healthy
    
    Returns:
        Dictionary with evaluation metrics
    """
    pred_damaged = predicted_damage < threshold
    true_damaged = true_damage < threshold
    
    # Calculate metrics
    tp = np.sum(pred_damaged & true_damaged)
    fp = np.sum(pred_damaged & ~true_damaged)
    tn = np.sum(~pred_damaged & ~true_damaged)
    fn = np.sum(~pred_damaged & true_damaged)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def plot_pso_results(history, predicted_damage, sensor_names, save_path=None):
    """Plot PSO optimization results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitness convergence
    ax = axes[0]
    ax.plot(history['global_best_scores'], label='Best Fitness', linewidth=2)
    ax.plot(history['mean_scores'], label='Mean Fitness', alpha=0.7)
    ax.fill_between(
        range(len(history['mean_scores'])),
        np.array(history['mean_scores']) - np.array(history['std_scores']),
        np.array(history['mean_scores']) + np.array(history['std_scores']),
        alpha=0.2
    )
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness (RMS Error)')
    ax.set_title('PSO Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Damage localization
    ax = axes[1]
    damage_severity = 1.0 - predicted_damage
    colors = ['red' if d > 0.1 else 'green' for d in damage_severity]
    bars = ax.bar(sensor_names, damage_severity, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Damage Threshold')
    ax.set_ylabel('Damage Severity (1 - Stiffness)')
    ax.set_title('Predicted Damage Localization')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, damage_severity):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pso_on_dataset(healthy_path, damage_paths, n_particles=50, n_iterations=50, 
                       max_samples=10, device='cuda'):
    """
    Run PSO on complete dataset
    
    Args:
        healthy_path: Path to healthy signal CSV
        damage_paths: List of paths to damaged signal CSVs
        n_particles: Number of PSO particles
        n_iterations: Number of PSO iterations
        max_samples: Maximum damage cases to process
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sensor_names = ['Sensor_24', 'Sensor_31', 'Sensor_32']
    
    # Load healthy signals
    print("Loading healthy signals...")
    healthy_signals_np = load_sensor_data(healthy_path, sensor_names)
    healthy_signals = prepare_signals_for_pso(healthy_signals_np).to(device)
    
    print(f"Healthy signals shape: {healthy_signals.shape}")
    
    results = []
    
    # Process damage cases
    print(f"\nProcessing up to {max_samples} damage cases...")
    for idx, damage_path in enumerate(damage_paths[:max_samples]):
        print(f"\n{'='*60}")
        print(f"Processing: {damage_path.split('/')[-2]}")
        print(f"{'='*60}")
        
        try:
            # Load damaged signals
            damaged_signals_np = load_sensor_data(damage_path, sensor_names)
            damaged_signals = prepare_signals_for_pso(damaged_signals_np).to(device)
            
            # Initialize PSO
            pso = BridgePSO(
                n_particles=n_particles,
                n_elements=len(sensor_names),
                bounds=(0.1, 1.0),
                c1=0.5,
                c2=0.3,
                w=0.9,
                device=device
            )
            
            # Run optimization
            best_damage = pso.optimize(
                healthy_signals,
                damaged_signals,
                n_iterations=n_iterations,
                verbose=True
            )
            
            # Convert to CPU for analysis
            predicted_damage = best_damage.cpu().numpy()
            
            # Store results
            result = {
                'damage_case': damage_path.split('/')[-2],
                'predicted_stiffness': predicted_damage.tolist(),
                'damage_severity': (1.0 - predicted_damage).tolist(),
                'final_fitness': pso.global_best_score,
                'sensor_names': sensor_names
            }
            results.append(result)
            
            # Plot results
            plot_pso_results(
                pso.history,
                predicted_damage,
                sensor_names,
                save_path=f"pso_result_{idx+1}.png"
            )
            
            print(f"\nResults for {result['damage_case']}:")
            for sensor, stiffness, damage in zip(sensor_names, predicted_damage, 1-predicted_damage):
                status = "DAMAGED" if damage > 0.1 else "HEALTHY"
                print(f"  {sensor}: Stiffness={stiffness:.3f}, Damage={damage:.3f} [{status}]")
            
        except Exception as e:
            print(f"Error processing {damage_path}: {e}")
            continue
    
    # Save all results
    results_df = pd.DataFrame([
        {
            'damage_case': r['damage_case'],
            **{f'{sensor}_stiffness': s for sensor, s in zip(sensor_names, r['predicted_stiffness'])},
            **{f'{sensor}_damage': d for sensor, d in zip(sensor_names, r['damage_severity'])},
            'final_fitness': r['final_fitness']
        }
        for r in results
    ])
    
    results_df.to_csv('pso_damage_localization_results.csv', index=False)
    print("\n" + "="*60)
    print("Results saved to: pso_damage_localization_results.csv")
    print("="*60)
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Paths (adjust for Kaggle environment)
    healthy_path = '/kaggle/input/dojo-dataset/complete_simulation_dataset/healthy/2017-11-03.csv'
    
    # Collect damage case paths
    damage_paths = []
    for i in range(1, 9):
        pattern = f'/kaggle/input/dojo-dataset/complete_simulation_dataset/damage_case_{i}_*/2017-11-03.csv'
        damage_paths.extend(glob(pattern))
    
    print(f"Found {len(damage_paths)} damage case files")
    
    # Run PSO
    results = run_pso_on_dataset(
        healthy_path=healthy_path,
        damage_paths=damage_paths,
        n_particles=50,
        n_iterations=50,
        max_samples=10,  # Process first 10 damage cases
        device='cuda'
    )
    
    print("\n✅ PSO optimization complete!")
    print(f"Processed {len(results)} damage cases")

if __name__ == "__main__":
    main()
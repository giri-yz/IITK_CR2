"""
GenAI Attack Generator
Uses trained VAE to generate novel attack scenarios by manipulating latent space
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.decomposition import PCA


class AttackGenerator:
    """Generate novel attack scenarios using VAE latent space manipulation"""
    
    def __init__(self, vae_model, normal_data: np.ndarray, feature_names: List[str]):
        self.vae = vae_model
        self.vae.eval()
        self.normal_data = torch.tensor(normal_data.astype(np.float32))
        self.feature_names = feature_names
        
        # Encode normal data to understand latent space
        with torch.no_grad():
            self.normal_mu, self.normal_logvar = self.vae.encode(self.normal_data)
        
        # Calculate latent space statistics
        self.latent_mean = self.normal_mu.mean(dim=0)
        self.latent_std = self.normal_mu.std(dim=0)
        
        print(f"✓ Attack Generator initialized")
        print(f"  Normal latent mean: {self.latent_mean.numpy()}")
        print(f"  Normal latent std: {self.latent_std.numpy()}")
    
    def generate_boundary_attack(self, sigma: float = 3.0, num_samples: int = 100) -> Dict:
        """
        Generate attacks by sampling from latent space boundaries
        (far from normal distribution)
        """
        attacks = []
        
        for _ in range(num_samples):
            # Sample from boundary of normal distribution
            z = self.latent_mean + sigma * self.latent_std * torch.randn(self.vae.latent_dim)
            
            # Decode to sensor space
            with torch.no_grad():
                attack_sample = self.vae.decode(z.unsqueeze(0)).squeeze()
            
            attacks.append(attack_sample.numpy())
        
        attacks = np.array(attacks)
        
        return {
            'type': 'boundary_exploration',
            'samples': attacks,
            'severity': 'HIGH',
            'description': f'Exploring {sigma}-sigma boundary of normal latent space',
            'expected_impact': 'Process deviation beyond normal operating range'
        }
    
    def generate_interpolation_attack(self, known_attack: np.ndarray, 
                                     num_steps: int = 20) -> Dict:
        """
        Generate attack variants by interpolating between normal and known attack
        Creates a spectrum of subtle to severe attacks
        """
        # Encode known attack
        attack_tensor = torch.tensor(known_attack.reshape(1, -1).astype(np.float32))
        with torch.no_grad():
            attack_mu, _ = self.vae.encode(attack_tensor)
        
        attacks = []
        alphas = np.linspace(0, 1, num_steps)
        
        for alpha in alphas:
            # Interpolate in latent space
            z = (1 - alpha) * self.latent_mean + alpha * attack_mu.squeeze()
            
            # Decode
            with torch.no_grad():
                interpolated = self.vae.decode(z.unsqueeze(0)).squeeze()
            
            attacks.append(interpolated.numpy())
        
        attacks = np.array(attacks)
        
        return {
            'type': 'interpolation_attack',
            'samples': attacks,
            'severity': 'VARIABLE',
            'description': f'Interpolated {num_steps} variants from normal to attack',
            'expected_impact': 'Gradual process degradation (tests detection threshold)'
        }
    
    def generate_targeted_sensor_attack(self, target_sensors: List[int], 
                                       magnitude: float = 0.3) -> Dict:
        """
        Generate attacks targeting specific sensors
        Manipulates only certain dimensions while keeping others normal
        """
        attacks = []
        
        for _ in range(50):
            # Start from normal state
            base_sample = self.normal_data[np.random.randint(len(self.normal_data))]
            attack_sample = base_sample.clone().numpy()
            
            # Manipulate target sensors
            for sensor_idx in target_sensors:
                # Add controlled perturbation
                attack_sample[sensor_idx] += magnitude * np.random.randn()
                # Clip to valid range [0, 1]
                attack_sample[sensor_idx] = np.clip(attack_sample[sensor_idx], 0, 1)
            
            attacks.append(attack_sample)
        
        attacks = np.array(attacks)
        target_names = [self.feature_names[i] for i in target_sensors]
        
        return {
            'type': 'targeted_sensor_manipulation',
            'targets': target_names,
            'samples': attacks,
            'severity': 'MEDIUM',
            'description': f'Manipulating sensors: {", ".join(target_names)}',
            'expected_impact': 'Localized process anomaly in specific subsystem'
        }
    
    def generate_coordinated_attack(self, sensor_groups: List[List[int]], 
                                   timing_offset: float = 0.1) -> Dict:
        """
        Generate coordinated multi-stage attacks
        Simulates sophisticated attack scenarios
        """
        attacks = []
        
        # Create attack sequence
        for stage_idx, sensor_group in enumerate(sensor_groups):
            stage_attacks = []
            
            for _ in range(20):
                base_sample = self.normal_data[np.random.randint(len(self.normal_data))]
                attack_sample = base_sample.clone().numpy()
                
                # Manipulate current stage sensors
                for sensor_idx in sensor_group:
                    perturbation = 0.2 * (stage_idx + 1) * np.random.randn()
                    attack_sample[sensor_idx] += perturbation
                    attack_sample[sensor_idx] = np.clip(attack_sample[sensor_idx], 0, 1)
                
                stage_attacks.append(attack_sample)
            
            attacks.extend(stage_attacks)
        
        attacks = np.array(attacks)
        
        return {
            'type': 'coordinated_multi_stage',
            'stages': len(sensor_groups),
            'samples': attacks,
            'severity': 'CRITICAL',
            'description': f'{len(sensor_groups)}-stage coordinated attack sequence',
            'expected_impact': 'Cascading process failure across multiple subsystems'
        }
    
    def generate_stealth_attack(self, epsilon: float = 0.05, num_samples: int = 100) -> Dict:
        """
        Generate subtle stealth attacks
        Small perturbations that might evade detection
        """
        attacks = []
        
        for _ in range(num_samples):
            # Start from normal
            base_sample = self.normal_data[np.random.randint(len(self.normal_data))]
            
            # Add small noise to latent representation
            with torch.no_grad():
                mu, logvar = self.vae.encode(base_sample.unsqueeze(0))
                # Small perturbation in latent space
                z = mu + epsilon * torch.randn_like(mu)
                stealth_sample = self.vae.decode(z).squeeze()
            
            attacks.append(stealth_sample.numpy())
        
        attacks = np.array(attacks)
        
        return {
            'type': 'stealth_perturbation',
            'samples': attacks,
            'severity': 'LOW',
            'description': f'Subtle attacks with epsilon={epsilon} latent perturbation',
            'expected_impact': 'Slow drift from normal operation (hard to detect)'
        }
    
    def analyze_attack_characteristics(self, attack_samples: np.ndarray) -> Dict:
        """
        Analyze characteristics of generated attacks
        """
        # Encode attacks to latent space
        attack_tensor = torch.tensor(attack_samples.astype(np.float32))
        with torch.no_grad():
            attack_mu, _ = self.vae.encode(attack_tensor)
        
        # Calculate distances from normal
        distances = torch.norm(attack_mu - self.latent_mean, dim=1)
        
        # Reconstruction errors
        with torch.no_grad():
            reconstructed = self.vae(attack_tensor)
            recon_errors = torch.mean((attack_tensor - reconstructed) ** 2, dim=1)
        
        # Feature-level deviations
        feature_deviations = np.abs(attack_samples - self.normal_data.numpy().mean(axis=0))
        max_deviations_per_feature = feature_deviations.max(axis=0)
        
        top_affected_features = np.argsort(max_deviations_per_feature)[-5:][::-1]
        
        return {
            'latent_distance_mean': float(distances.mean()),
            'latent_distance_std': float(distances.std()),
            'recon_error_mean': float(recon_errors.mean()),
            'recon_error_std': float(recon_errors.std()),
            'most_affected_sensors': [
                {'name': self.feature_names[i], 
                 'max_deviation': float(max_deviations_per_feature[i])}
                for i in top_affected_features
            ]
        }
    
    def generate_attack_suite(self) -> List[Dict]:
        """
        Generate comprehensive suite of attack scenarios for testing
        """
        print("\n🧪 Generating comprehensive attack suite...")
        
        suite = []
        
        # 1. Boundary attacks
        print("  → Generating boundary attacks...")
        suite.append(self.generate_boundary_attack(sigma=2.0, num_samples=50))
        suite.append(self.generate_boundary_attack(sigma=3.5, num_samples=50))
        
        # 2. Stealth attacks
        print("  → Generating stealth attacks...")
        suite.append(self.generate_stealth_attack(epsilon=0.03, num_samples=100))
        suite.append(self.generate_stealth_attack(epsilon=0.08, num_samples=100))
        
        # 3. Targeted sensor attacks (common critical sensors)
        print("  → Generating targeted sensor attacks...")
        suite.append(self.generate_targeted_sensor_attack([0, 1], magnitude=0.25))  # Flow + Tank
        suite.append(self.generate_targeted_sensor_attack([4, 5], magnitude=0.30))  # Pressure sensors
        suite.append(self.generate_targeted_sensor_attack([8, 9], magnitude=0.20))  # Pumps
        
        # 4. Coordinated attacks
        print("  → Generating coordinated attacks...")
        suite.append(self.generate_coordinated_attack([[0, 1], [4, 5], [8, 9]]))
        
        print(f"✓ Generated {len(suite)} attack scenarios")
        
        return suite


# Feature names for SWaT (example - adjust based on your actual features)
SWAT_FEATURES = [
    'FIT101_Flow',
    'LIT101_Level', 
    'FIT201_Flow',
    'AIT201_Conductivity',
    'FIT301_Flow',
    'LIT301_Level',
    'FIT401_Flow',
    'LIT401_Level',
    'P101_Pump',
    'P201_Pump',
    'P301_Pump',
    'P401_Pump',
    'MV101_Valve',
    'MV201_Valve'
]


if __name__ == "__main__":
    print("Attack Generator Test")
    print("Load your VAE model and normal data, then instantiate AttackGenerator")
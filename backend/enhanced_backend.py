"""
Updated Flask Backend Integration
Replaces simple VAE detection with enhanced multi-layer system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Import enhanced detection
from enhanced_detection_system import EnhancedDetectionSystem


# ======================
# VAE Model Definition
# ======================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)


# ======================
# Global Setup
# ======================
app = Flask(__name__)
CORS(app)

# Load model and data
MODEL_PATH = "vae_model.pth"  # Update path
NORMAL_DATA_PATH = "clean_swat.csv"  # Update path
THRESHOLD = 0.118232

print("🚀 Loading VAE model...")
model = VAE(14, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

print("📊 Loading normal data baseline...")
normal_df = pd.read_csv(NORMAL_DATA_PATH)
normal_data = normal_df.values.astype(np.float32)

print("🛡️ Initializing Enhanced Detection System...")
detector = EnhancedDetectionSystem(model, THRESHOLD, normal_data)

print("✅ Backend ready!")


# ======================
# API Endpoints
# ======================
@app.route('/detect', methods=['POST'])
def detect():
    """
    ENHANCED detection endpoint with multi-layer analysis
    """
    try:
        data = request.json
        sensor_data = np.array(data.get('sensor_data', []))
        
        if len(sensor_data) != 14:
            return jsonify({
                'error': 'Expected 14 sensor values',
                'received': len(sensor_data)
            }), 400
        
        # Run enhanced detection
        result = detector.ensemble_detect(sensor_data)
        
        # Format response
        return jsonify({
            'attack_detected': result['attack_detected'],
            'reconstruction_error': result['reconstruction_error'],
            'confidence': result['confidence'],
            'severity': result['severity'],
            'status': 'ATTACK DETECTED' if result['attack_detected'] else 'NORMAL',
            
            # Detailed breakdown
            'detection_breakdown': {
                'vae_detected': result['vote_breakdown']['vae'],
                'temporal_detected': result['vote_breakdown']['temporal'],
                'correlation_detected': result['vote_breakdown']['correlation'],
                'physics_detected': result['vote_breakdown']['physics'],
                'votes': result['vote_count'],
                'votes_needed': 2
            },
            
            # Thresholds used
            'thresholds': {
                'original': THRESHOLD,
                'adaptive_95': detector.threshold_95,
                'adaptive_mean_2std': detector.threshold_adaptive
            },
            
            # Explanation
            'explanation': result['explanation'],
            
            # Detailed layer results (for debugging)
            'layer_details': {
                'vae': {
                    'error': result['detection_reasons']['vae']['error'],
                    'confidence': result['detection_reasons']['vae']['confidence'],
                    'threshold_votes': result['detection_reasons']['vae']['threshold_votes'],
                    'anomalous_features': result['detection_reasons']['vae'].get('anomalous_features', [])
                },
                'temporal': result['detection_reasons']['temporal'],
                'correlation': result['detection_reasons']['correlation'],
                'physics': result['detection_reasons']['physics']
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'attack_detected': False
        }), 500


@app.route('/detect/reset', methods=['POST'])
def reset_detection():
    """Reset temporal history buffer"""
    try:
        detector.reset_history()
        return jsonify({
            'success': True,
            'message': 'Detection history cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect/calibrate', methods=['POST'])
def calibrate_thresholds():
    """
    Recalibrate detection thresholds based on new normal data
    """
    try:
        data = request.json
        new_normal_samples = np.array(data.get('normal_samples', []))
        
        if len(new_normal_samples) == 0:
            return jsonify({'error': 'No samples provided'}), 400
        
        # Recalculate thresholds
        with torch.no_grad():
            normal_tensor = torch.tensor(new_normal_samples.astype(np.float32))
            normal_recon = model(normal_tensor)
            errors = torch.mean((normal_tensor - normal_recon)**2, dim=1).numpy()
        
        old_threshold_95 = detector.threshold_95
        
        detector.threshold_95 = np.percentile(errors, 95)
        detector.threshold_99 = np.percentile(errors, 99)
        detector.threshold_mean = np.mean(errors)
        detector.threshold_std = np.std(errors)
        detector.threshold_adaptive = detector.threshold_mean + 2 * detector.threshold_std
        
        return jsonify({
            'success': True,
            'message': 'Thresholds recalibrated',
            'old_threshold_95': float(old_threshold_95),
            'new_thresholds': {
                'threshold_95': float(detector.threshold_95),
                'threshold_99': float(detector.threshold_99),
                'threshold_adaptive': float(detector.threshold_adaptive),
                'mean': float(detector.threshold_mean),
                'std': float(detector.threshold_std)
            },
            'samples_used': len(new_normal_samples)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect/stats', methods=['GET'])
def detection_stats():
    """Get detection system statistics"""
    return jsonify({
        'thresholds': {
            'original': THRESHOLD,
            'threshold_95': float(detector.threshold_95),
            'threshold_99': float(detector.threshold_99),
            'adaptive': float(detector.threshold_adaptive),
            'mean_normal_error': float(detector.threshold_mean),
            'std_normal_error': float(detector.threshold_std)
        },
        'history_buffer_size': len(detector.history_buffer),
        'detection_layers': ['VAE', 'Temporal', 'Correlation', 'Physics'],
        'voting_threshold': 2,
        'confidence_boost_on_critical': True
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'detector_ready': detector is not None,
        'detection_system': 'enhanced_multi_layer'
    })


# ======================
# Run Server
# ======================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🛡️ Enhanced Detection Backend Running")
    print("="*70)
    print("\nDetection Layers:")
    print("  ✓ Layer 1: Multi-threshold VAE")
    print("  ✓ Layer 2: Temporal pattern analysis")
    print("  ✓ Layer 3: Correlation checks")
    print("  ✓ Layer 4: Physics validation")
    print("  ✓ Layer 5: Ensemble voting")
    print("\nThresholds configured:")
    print(f"  • 95th percentile: {detector.threshold_95:.6f}")
    print(f"  • Adaptive (μ+2σ): {detector.threshold_adaptive:.6f}")
    print(f"  • Original:        {THRESHOLD:.6f}")
    print("="*70)
    print("\nStarting Flask server on http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

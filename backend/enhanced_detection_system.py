"""
Enhanced Multi-Layer Detection System for GenTwin
==================================================
Addresses the 0% detection rate issue by adding multiple detection layers:
1. Improved VAE with better threshold calibration
2. Temporal pattern analysis (sequence-based detection)
3. Statistical anomaly detection
4. Physics-based correlation checks
5. Ensemble voting system
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from scipy import stats


class EnhancedDetectionSystem:
    """
    Multi-layer detection system that catches subtle attacks
    """
    
    def __init__(self, vae_model, threshold: float, normal_data: np.ndarray):
        self.vae = vae_model
        self.vae.eval()
        self.base_threshold = threshold
        
        # Calibrate better thresholds from normal data
        with torch.no_grad():
            normal_tensor = torch.tensor(normal_data.astype(np.float32))
            normal_recon = self.vae(normal_tensor)
            self.normal_errors = torch.mean((normal_tensor - normal_recon)**2, dim=1).numpy()
        
        # Calculate adaptive thresholds
        self.threshold_mean = np.mean(self.normal_errors)
        self.threshold_std = np.std(self.normal_errors)
        
        # Multiple threshold levels
        self.threshold_95 = np.percentile(self.normal_errors, 95)  # Stricter
        self.threshold_99 = np.percentile(self.normal_errors, 99)  # More lenient
        self.threshold_adaptive = self.threshold_mean + 2 * self.threshold_std
        
        print(f"📊 Enhanced Detection Initialized:")
        print(f"   Original Threshold: {threshold:.6f}")
        print(f"   95th Percentile:    {self.threshold_95:.6f}")
        print(f"   99th Percentile:    {self.threshold_99:.6f}")
        print(f"   Adaptive (μ+2σ):    {self.threshold_adaptive:.6f}")
        print(f"   Mean Normal Error:  {self.threshold_mean:.6f}")
        
        # Temporal pattern buffer
        self.history_buffer = deque(maxlen=10)
        
        # Statistical baselines
        self.normal_mean = np.mean(normal_data, axis=0)
        self.normal_std = np.std(normal_data, axis=0)
        self.normal_correlations = np.corrcoef(normal_data.T)
        
        # Feature importance (which sensors matter most)
        self.feature_sensitivity = self._calculate_feature_sensitivity(normal_data)
    
    def _calculate_feature_sensitivity(self, normal_data: np.ndarray) -> np.ndarray:
        """Calculate which features are most sensitive to changes"""
        # Use coefficient of variation (std/mean) as sensitivity measure
        cv = self.normal_std / (self.normal_mean + 1e-6)
        # Normalize to 0-1
        sensitivity = (cv - cv.min()) / (cv.max() - cv.min() + 1e-6)
        return sensitivity
    
    def detect_vae_anomaly(self, sensor_data: np.ndarray) -> Dict:
        """
        LAYER 1: Improved VAE detection with multiple thresholds
        """
        x = torch.tensor([sensor_data], dtype=torch.float32)
        
        with torch.no_grad():
            recon = self.vae(x)
            error = torch.mean((x - recon)**2).item()
            
            # Per-feature errors (which sensors are most anomalous)
            feature_errors = ((x - recon)**2).squeeze().numpy()
            weighted_errors = feature_errors * self.feature_sensitivity
            max_weighted_error = np.max(weighted_errors)
        
        # Multi-threshold voting
        detected_95 = error > self.threshold_95
        detected_99 = error > self.threshold_99
        detected_adaptive = error > self.threshold_adaptive
        detected_original = error > self.base_threshold
        
        # If ANY threshold triggers, consider it suspicious
        detected = detected_95 or detected_adaptive
        
        # Confidence based on how many thresholds triggered
        confidence = sum([detected_95, detected_99, detected_adaptive, detected_original]) / 4.0
        
        return {
            'detected': detected,
            'error': error,
            'confidence': confidence,
            'threshold_votes': {
                '95th_percentile': detected_95,
                '99th_percentile': detected_99,
                'adaptive': detected_adaptive,
                'original': detected_original
            },
            'max_weighted_feature_error': float(max_weighted_error),
            'anomalous_features': np.where(weighted_errors > 0.1)[0].tolist()
        }
    
    def detect_temporal_anomaly(self, sensor_data: np.ndarray) -> Dict:
        """
        LAYER 2: Temporal pattern analysis
        Detects sudden changes or drift
        """
        self.history_buffer.append(sensor_data)
        
        if len(self.history_buffer) < 3:
            return {'detected': False, 'reason': 'insufficient_history'}
        
        history = np.array(list(self.history_buffer))
        
        # Check for sudden jumps
        if len(history) >= 2:
            recent_change = np.abs(history[-1] - history[-2])
            max_change = np.max(recent_change)
            
            # Historical change distribution
            if len(history) >= 5:
                historical_changes = np.diff(history, axis=0)
                typical_change = np.mean(np.abs(historical_changes), axis=0)
                change_ratio = recent_change / (typical_change + 1e-6)
                
                # Flag if change is > 3x typical
                sudden_jump = np.any(change_ratio > 3.0)
                
                if sudden_jump:
                    return {
                        'detected': True,
                        'reason': 'sudden_jump',
                        'max_change_ratio': float(np.max(change_ratio)),
                        'affected_sensors': np.where(change_ratio > 3.0)[0].tolist()
                    }
        
        # Check for drift (gradual deviation from baseline)
        current_deviation = np.abs(sensor_data - self.normal_mean) / (self.normal_std + 1e-6)
        drift_detected = np.any(current_deviation > 3.0)  # > 3 standard deviations
        
        if drift_detected:
            return {
                'detected': True,
                'reason': 'statistical_drift',
                'max_deviation': float(np.max(current_deviation)),
                'drifting_sensors': np.where(current_deviation > 3.0)[0].tolist()
            }
        
        return {'detected': False, 'reason': 'normal_temporal_pattern'}
    
    def detect_correlation_anomaly(self, sensor_data: np.ndarray) -> Dict:
        """
        LAYER 3: Correlation-based detection
        Some sensors should correlate - attacks often break these relationships
        """
        # Check critical correlations
        # Example: Flow and Level should correlate (P1: FIT101 and LIT101)
        critical_pairs = [
            (0, 1),   # FIT101 - LIT101
            (2, 3),   # FIT201 - AIT201
            (4, 5),   # FIT301 - LIT301
            (6, 7),   # FIT401 - LIT401
        ]
        
        violations = []
        
        for i, j in critical_pairs:
            # Get expected correlation
            expected_corr = self.normal_correlations[i, j]
            
            # If sensors should be positively correlated
            if expected_corr > 0.5:
                # Check if they're moving in opposite directions
                deviation_i = sensor_data[i] - self.normal_mean[i]
                deviation_j = sensor_data[j] - self.normal_mean[j]
                
                # They should have same sign
                if np.sign(deviation_i) != np.sign(deviation_j):
                    if abs(deviation_i) > 0.15 or abs(deviation_j) > 0.15:
                        violations.append({
                            'sensors': (i, j),
                            'expected_corr': float(expected_corr),
                            'issue': 'opposite_directions'
                        })
        
        # Check for impossible physical states
        # Example: High flow with very low tank level shouldn't happen
        flow_high = sensor_data[0] > 0.8
        level_low = sensor_data[1] < 0.3
        
        if flow_high and level_low:
            violations.append({
                'sensors': (0, 1),
                'issue': 'impossible_state',
                'description': 'High inflow with critically low tank level'
            })
        
        detected = len(violations) > 0
        
        return {
            'detected': detected,
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def detect_physics_violations(self, sensor_data: np.ndarray) -> Dict:
        """
        LAYER 4: Enhanced physics-based checks
        """
        violations = []
        
        # Critical Level Checks (more sensitive)
        if sensor_data[1] > 0.85:  # LIT101 > 85%
            violations.append({
                'type': 'CRITICAL_LEVEL',
                'sensor': 'LIT101',
                'value': float(sensor_data[1]),
                'threshold': 0.85,
                'severity': 'HIGH'
            })
        
        if sensor_data[1] < 0.25:  # LIT101 < 25%
            violations.append({
                'type': 'CRITICAL_LEVEL',
                'sensor': 'LIT101',
                'value': float(sensor_data[1]),
                'threshold': 0.25,
                'severity': 'HIGH'
            })
        
        # Flow-Pump Consistency
        # If pump is OFF (< 0.3) but flow is HIGH (> 0.7), something's wrong
        if sensor_data[8] < 0.3 and sensor_data[0] > 0.7:
            violations.append({
                'type': 'FLOW_PUMP_MISMATCH',
                'description': 'High flow with pump off',
                'severity': 'MEDIUM'
            })
        
        # Conductivity bounds (more strict)
        if sensor_data[3] < 0.15 or sensor_data[3] > 0.85:
            violations.append({
                'type': 'CONDUCTIVITY_ANOMALY',
                'value': float(sensor_data[3]),
                'severity': 'MEDIUM'
            })
        
        # Multi-tank coordination
        # All tanks shouldn't be at extremes simultaneously
        tank_levels = [sensor_data[1], sensor_data[5], sensor_data[7]]
        if all(level > 0.85 for level in tank_levels):
            violations.append({
                'type': 'SYSTEM_WIDE_ANOMALY',
                'description': 'All tanks near overflow simultaneously',
                'severity': 'CRITICAL'
            })
        
        detected = len(violations) > 0
        
        return {
            'detected': detected,
            'violations': violations,
            'severity': max([v.get('severity', 'LOW') for v in violations], default='NONE')
        }
    
    def ensemble_detect(self, sensor_data: np.ndarray) -> Dict:
        """
        FINAL LAYER: Ensemble voting across all detection methods
        """
        # Run all detection layers
        vae_result = self.detect_vae_anomaly(sensor_data)
        temporal_result = self.detect_temporal_anomaly(sensor_data)
        correlation_result = self.detect_correlation_anomaly(sensor_data)
        physics_result = self.detect_physics_violations(sensor_data)
        
        # Collect votes
        votes = {
            'vae': vae_result['detected'],
            'temporal': temporal_result['detected'],
            'correlation': correlation_result['detected'],
            'physics': physics_result['detected']
        }
        
        vote_count = sum(votes.values())
        
        # Detection logic: 
        # - 2+ votes = DETECTED
        # - Physics CRITICAL = DETECTED
        # - VAE high confidence (>0.75) = DETECTED
        
        detected = (
            vote_count >= 2 or
            (physics_result.get('severity') == 'CRITICAL') or
            (vae_result['confidence'] > 0.75)
        )
        
        # Calculate overall confidence
        confidence = vote_count / 4.0
        if physics_result.get('severity') == 'CRITICAL':
            confidence = max(confidence, 0.9)
        
        # Determine attack severity
        if vae_result['error'] > 0.15:
            severity = 'EXTREME'
        elif vae_result['error'] > 0.05:
            severity = 'MEDIUM'
        else:
            severity = 'SUBTLE'
        
        return {
            'attack_detected': detected,
            'confidence': confidence,
            'severity': severity,
            'vote_breakdown': votes,
            'vote_count': vote_count,
            'detection_reasons': {
                'vae': vae_result,
                'temporal': temporal_result,
                'correlation': correlation_result,
                'physics': physics_result
            },
            'reconstruction_error': vae_result['error'],
            'explanation': self._generate_explanation(detected, votes, vae_result, physics_result)
        }
    
    def _generate_explanation(self, detected: bool, votes: Dict, 
                             vae_result: Dict, physics_result: Dict) -> str:
        """Generate human-readable explanation"""
        if not detected:
            return "All detection layers report normal operation"
        
        reasons = []
        if votes['vae']:
            reasons.append(f"VAE anomaly (error={vae_result['error']:.4f}, {vae_result['confidence']*100:.0f}% confidence)")
        if votes['temporal']:
            reasons.append("Temporal pattern anomaly detected")
        if votes['correlation']:
            reasons.append("Sensor correlation violated")
        if votes['physics']:
            reasons.append(f"Physics violations ({physics_result.get('severity', 'UNKNOWN')} severity)")
        
        return " | ".join(reasons)
    
    def reset_history(self):
        """Clear temporal history buffer"""
        self.history_buffer.clear()


# ======================
# Integration Example
# ======================
if __name__ == "__main__":
    print("Enhanced Detection System - Standalone Test")
    print("="*70)
    print("\nThis system adds multiple detection layers:")
    print("  1. Improved VAE with adaptive thresholds")
    print("  2. Temporal pattern analysis")
    print("  3. Correlation-based detection")
    print("  4. Enhanced physics checks")
    print("  5. Ensemble voting")
    print("\nIntegrate this into your Flask backend to replace simple VAE detection.")
    print("="*70)

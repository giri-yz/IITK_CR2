import React, { useEffect, useState } from 'react';
import './AnomalyMeter.css';

const AnomalyMeter = ({ vaeResult, threshold = 0.118232 }) => {
  const [score, setScore] = useState(0);
  const [isAttack, setIsAttack] = useState(false);
  
  useEffect(() => {
    if (vaeResult && vaeResult.reconstruction_error !== undefined) {
      setScore(vaeResult.reconstruction_error);
      setIsAttack(vaeResult.attack_detected || false);
    }
  }, [vaeResult]);
  
  // Calculate percentage for visual bar (cap at 0.3 for display)
  const maxDisplay = 0.3;
  const percentage = Math.min((score / maxDisplay) * 100, 100);
  const thresholdPercentage = (threshold / maxDisplay) * 100;
  
  return (
    <div className="anomaly-meter">
      <div className="meter-header">
        <div className="meter-title">
          <h3>Reconstruction Error</h3>
          <span className="meter-subtitle">VAE Anomaly Detection</span>
        </div>
        <div className="meter-value-display">
          <span className="meter-value">{score.toFixed(4)}</span>
          <span className="meter-unit">error</span>
        </div>
      </div>
      
      <div className="meter-bar-container">
        <div className="meter-bar">
          <div 
            className={`meter-fill ${isAttack ? 'meter-fill--attack' : 'meter-fill--normal'}`}
            style={{ width: `${percentage}%` }}
          >
            <div className="meter-glow"></div>
          </div>
          
          <div 
            className="threshold-line" 
            style={{ left: `${thresholdPercentage}%` }}
            title={`Threshold: ${threshold.toFixed(4)}`}
          >
            <div className="threshold-label">T</div>
          </div>
        </div>
        
        <div className="meter-scale">
          <span>0.000</span>
          <span>0.150</span>
          <span>0.300</span>
        </div>
      </div>
      
      <div className="meter-status">
        <div className={`status-indicator ${isAttack ? 'status-indicator--attack' : 'status-indicator--normal'}`}>
          <div className="status-dot"></div>
          <span className="status-text">
            {isAttack ? '🚨 ANOMALY DETECTED' : '✓ Normal Operation'}
          </span>
        </div>
        
        <div className="meter-details">
          <div className="detail-item">
            <span className="detail-label">Threshold:</span>
            <span className="detail-value">{threshold.toFixed(4)}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Status:</span>
            <span className={`detail-value ${isAttack ? 'text-danger' : 'text-success'}`}>
              {vaeResult?.status || 'READY'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnomalyMeter;
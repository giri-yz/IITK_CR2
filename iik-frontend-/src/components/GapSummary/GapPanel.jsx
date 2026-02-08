import React, { useState, useEffect } from 'react';
import { discoverGaps } from '../../services/api';
import './GapPanel.css';

const GapPanel = ({ attackedStage, attackType, attackPattern }) => {
  const [gaps, setGaps] = useState([]);
  const [isDiscovering, setIsDiscovering] = useState(false);

  // Discover gaps using AI when attack happens
  useEffect(() => {
    if (attackedStage && attackType && attackPattern) {
      setTimeout(async () => {
        await discoverGapsWithAI(attackPattern);
      }, 7500);
    }
  }, [attackedStage, attackType, attackPattern]);

  const discoverGapsWithAI = async (pattern) => {
    setIsDiscovering(true);
    
    try {
      console.log('🔍 Discovering gaps with AI...');
      
      // Call backend AI gap discovery
      const discoveredGaps = await discoverGaps(pattern);
      
      // Convert to UI format
      const formattedGaps = discoveredGaps.map((gap, index) => ({
        id: Date.now() + index,
        stage: getStageName(attackedStage),
        vulnerability: gap.description,
        severity: gap.severity,
        riskScore: getSeverityScore(gap.severity),
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
        mitigationSteps: [gap.recommendation]
      }));
      
      setGaps(prev => [...formattedGaps, ...prev].slice(0, 5));
      
    } catch (error) {
      console.error('Failed to discover gaps:', error);
    } finally {
      setIsDiscovering(false);
    }
  };

  const getStageName = (stageId) => {
    const names = {
      stage1: 'Raw Water Intake',
      stage2: 'Chemical Dosing',
      stage3: 'Filtration',
      stage4: 'UV Treatment',
      stage5: 'pH Adjustment',
      stage6: 'Clean Water Storage'
    };
    return names[stageId] || stageId;
  };

  const getSeverityScore = (severity) => {
    const scores = {
      'critical': 9.5,
      'high': 8.0,
      'medium': 6.5,
      'low': 4.0
    };
    return scores[severity] || 7.0;
  };

  const clearGaps = () => {
    setGaps([]);
  };

  return (
    <div className="gap-panel">
      <div className="gap-header">
        <div className="gap-title">
          <h3>Cybersecurity Gaps</h3>
          <span className="gap-count">
            {isDiscovering ? 'Discovering...' : `${gaps.length} discovered`}
          </span>
        </div>
        {gaps.length > 0 && (
          <button className="clear-button" onClick={clearGaps} title="Clear all gaps">
            🗑️
          </button>
        )}
      </div>

      <div className="gap-list">
        {gaps.length === 0 ? (
          <div className="gap-empty">
            <div className="empty-icon">🛡️</div>
            <p className="empty-text">
              {isDiscovering ? 'AI analyzing vulnerabilities...' : 'No gaps discovered yet'}
            </p>
            <p className="empty-subtitle">
              {isDiscovering ? 'Using digital twin simulation' : 'Inject an attack to discover vulnerabilities'}
            </p>
          </div>
        ) : (
          gaps.map(gap => (
            <div key={gap.id} className="gap-card">
              <div className="gap-card-header">
                <div className="gap-stage">
                  <span className="stage-icon">⚙️</span>
                  <span className="stage-name">{gap.stage}</span>
                </div>
                <span className={`gap-severity gap-severity--${gap.severity}`}>
                  {gap.severity}
                </span>
              </div>

              <div className="gap-vulnerability">
                <span className="vulnerability-icon">🔓</span>
                <span className="vulnerability-text">{gap.vulnerability}</span>
              </div>

              <div className="gap-risk">
                <div className="risk-label-row">
                  <span className="risk-label">Risk Score</span>
                  <span className="risk-value">{gap.riskScore}/10</span>
                </div>
                <div className="risk-bar">
                  <div 
                    className={`risk-fill risk-fill--${gap.severity}`}
                    style={{ width: `${gap.riskScore * 10}%` }}
                  ></div>
                </div>
              </div>

              <div className="gap-mitigation">
                <span className="mitigation-label">AI Recommendation:</span>
                <ul className="mitigation-steps">
                  {gap.mitigationSteps.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ul>
              </div>

              <div className="gap-footer">
                <span className="gap-timestamp">Discovered: {gap.timestamp}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default GapPanel;
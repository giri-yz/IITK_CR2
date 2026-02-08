import React, { useState, useEffect } from 'react';
import './PlantDiagram.css';

const PlantDiagram = ({ attackedStage, attackType }) => {
  const [waterLevels, setWaterLevels] = useState({
    stage1: 75,
    stage2: 68,
    stage3: 82,
    stage4: 71,
    stage5: 79,
    stage6: 85
  });

  const [pumpStates, setPumpStates] = useState({
    P101: true,
    P201: true,
    P301: true,
    P401: true,
    P501: true,
    P601: true
  });

  const [flowRates, setFlowRates] = useState({
    stage1: 2.4,
    stage2: 2.3,
    stage3: 2.5,
    stage4: 2.2,
    stage5: 2.4,
    stage6: 2.1
  });

  // Simulate attack effects on water levels
  useEffect(() => {
    if (attackedStage && attackType) {
      const interval = setInterval(() => {
        setWaterLevels(prev => ({
          ...prev,
          [attackedStage]: Math.min(95, prev[attackedStage] + 2)
        }));
      }, 200);

      setTimeout(() => clearInterval(interval), 4000);
      return () => clearInterval(interval);
    }
  }, [attackedStage, attackType]);

  const stages = [
    { 
      id: 'stage1', 
      name: 'Raw Water Intake',
      subtitle: 'Primary Filtration',
      tank: 'T-101',
      sensor: 'LIT-101',
      pump: 'P-101',
      valve: 'MV-101',
      icon: '💧',
      color: '#3B82F6'
    },
    { 
      id: 'stage2', 
      name: 'Chemical Dosing',
      subtitle: 'Coagulation Process',
      tank: 'T-201',
      sensor: 'LIT-201',
      pump: 'P-201',
      valve: 'MV-201',
      icon: '⚗️',
      color: '#8B5CF6'
    },
    { 
      id: 'stage3', 
      name: 'Filtration',
      subtitle: 'Sand & Carbon Filter',
      tank: 'T-301',
      sensor: 'LIT-301',
      pump: 'P-301',
      valve: 'MV-301',
      icon: '🔬',
      color: '#06B6D4'
    },
    { 
      id: 'stage4', 
      name: 'UV Treatment',
      subtitle: 'Disinfection',
      tank: 'T-401',
      sensor: 'LIT-401',
      pump: 'P-401',
      valve: 'MV-401',
      icon: '☀️',
      color: '#F59E0B'
    },
    { 
      id: 'stage5', 
      name: 'pH Adjustment',
      subtitle: 'Chemical Balance',
      tank: 'T-501',
      sensor: 'LIT-501',
      pump: 'P-501',
      valve: 'MV-501',
      icon: '⚖️',
      color: '#10B981'
    },
    { 
      id: 'stage6', 
      name: 'Clean Water Storage',
      subtitle: 'Distribution Ready',
      tank: 'T-601',
      sensor: 'LIT-601',
      pump: 'P-601',
      valve: 'MV-601',
      icon: '✓',
      color: '#4ADE80'
    }
  ];

  return (
    <div className="plant-diagram-container">
      {/* Header */}
      <div className="plant-header">
        <div className="header-content">
          <div className="header-icon">🏭</div>
          <div className="header-text">
            <h2>Industrial Water Treatment Plant</h2>
            <p>6-Stage Continuous Flow System • SCADA Monitor</p>
          </div>
        </div>
        <div className="system-metrics">
          <div className="metric-item">
            <span className="metric-label">Total Flow</span>
            <span className="metric-value">14.9 m³/h</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">System Status</span>
            <span className={`metric-value ${attackedStage ? 'status-critical' : 'status-normal'}`}>
              {attackedStage ? '⚠️ ALERT' : '✓ NORMAL'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Process Flow */}
      <div className="process-flow">
        {stages.map((stage, index) => {
          const isAttacked = attackedStage === stage.id;
          const level = waterLevels[stage.id];
          const flowRate = flowRates[stage.id];
          const pumpActive = pumpStates[stage.pump.replace('-', '')];

          return (
            <React.Fragment key={stage.id}>
              {/* Process Stage */}
              <div className={`process-stage ${isAttacked ? 'stage-attacked' : ''}`}>
                {/* Stage Header */}
                <div className="stage-header">
                  <div className="stage-info">
                    <span className="stage-icon">{stage.icon}</span>
                    <div className="stage-text">
                      <h3>{stage.name}</h3>
                      <p>{stage.subtitle}</p>
                    </div>
                  </div>
                  {isAttacked && (
                    <div className="attack-indicator">
                      <span className="attack-pulse"></span>
                      ATTACK DETECTED
                    </div>
                  )}
                </div>

                {/* 3D Industrial Tank */}
                <div className="industrial-tank">
                  {/* Tank Top Cap */}
                  <div className="tank-cap">
                    <div className="cap-label">{stage.tank}</div>
                    <div className="vent-pipe"></div>
                  </div>

                  {/* Tank Body - 3D Cylinder */}
                  <div className={`tank-body ${isAttacked ? 'tank-attacked' : ''}`}>
                    {/* Water Fill with Realistic Physics */}
                    <div 
                      className={`water-fill ${isAttacked ? 'water-turbulent' : ''}`}
                      style={{ 
                        height: `${level}%`,
                        background: isAttacked 
                          ? 'linear-gradient(180deg, rgba(239,68,68,0.9) 0%, rgba(220,38,38,1) 100%)'
                          : `linear-gradient(180deg, ${stage.color}CC 0%, ${stage.color} 100%)`
                      }}
                    >
                      {/* Water Surface Animation */}
                      <div className="water-surface">
                        <div className="wave wave-1"></div>
                        <div className="wave wave-2"></div>
                        <div className="wave wave-3"></div>
                      </div>

                      {/* Bubbles (for aeration) */}
                      {index === 2 && (
                        <div className="bubbles-container">
                          {[...Array(8)].map((_, i) => (
                            <div 
                              key={i} 
                              className="bubble"
                              style={{ 
                                left: `${15 + i * 10}%`,
                                animationDelay: `${i * 0.3}s`,
                                animationDuration: `${2 + Math.random() * 2}s`
                              }}
                            />
                          ))}
                        </div>
                      )}

                      {/* UV Light Effect */}
                      {index === 3 && !isAttacked && (
                        <div className="uv-light-effect"></div>
                      )}
                    </div>

                    {/* Tank Glass Reflection */}
                    <div className="tank-reflection"></div>

                    {/* Level Measurement Lines */}
                    <div className="level-markers">
                      <div className="marker marker-100"><span>100%</span></div>
                      <div className="marker marker-75"><span>75%</span></div>
                      <div className="marker marker-50"><span>50%</span></div>
                      <div className="marker marker-25"><span>25%</span></div>
                    </div>

                    {/* Digital Level Display */}
                    <div className="level-display">
                      <span className="level-number">{level}%</span>
                      <span className="level-mm">{Math.round(level * 10)}mm</span>
                    </div>
                  </div>

                  {/* Tank Base */}
                  <div className="tank-base">
                    <div className="base-outlet"></div>
                  </div>
                </div>

                {/* Control Panel */}
                <div className="control-panel">
                  {/* Sensor Reading */}
                  <div className="control-section">
                    <div className="control-label">
                      <span className="control-icon">📊</span>
                      {stage.sensor}
                    </div>
                    <div className="control-display">
                      <span className="display-value">{level}%</span>
                      <div className={`status-led ${isAttacked ? 'led-critical' : 'led-normal'}`}></div>
                    </div>
                  </div>

                  {/* Pump Control */}
                  <div className="control-section">
                    <div className="control-label">
                      <span className="control-icon">⚡</span>
                      {stage.pump}
                    </div>
                    <div className="control-display">
                      <span className="display-value">{pumpActive ? 'ON' : 'OFF'}</span>
                      <div className={`status-led ${pumpActive ? 'led-active' : 'led-inactive'}`}></div>
                    </div>
                  </div>

                  {/* Flow Rate */}
                  <div className="control-section">
                    <div className="control-label">
                      <span className="control-icon">💨</span>
                      Flow Rate
                    </div>
                    <div className="control-display">
                      <span className="display-value">{flowRate} m³/h</span>
                      <div className="status-led led-flow"></div>
                    </div>
                  </div>

                  {/* Valve Control */}
                  <div className="control-section">
                    <div className="control-label">
                      <span className="control-icon">🔧</span>
                      {stage.valve}
                    </div>
                    <div className="control-display">
                      <span className="display-value">45%</span>
                      <div className="valve-indicator">
                        <div className="valve-bar" style={{ width: '45%' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Connecting Pipe (except after last stage) */}
              {index < stages.length - 1 && (
                <div className="connecting-pipe">
                  <div className="pipe-horizontal">
                    {/* Animated water flow */}
                    <div className={`flow-animation ${isAttacked ? 'flow-disrupted' : ''}`}>
                      {[...Array(6)].map((_, i) => (
                        <div 
                          key={i} 
                          className="flow-particle"
                          style={{ 
                            animationDelay: `${i * 0.4}s`,
                            backgroundColor: isAttacked && i > 2 ? '#EF4444' : stage.color
                          }}
                        />
                      ))}
                    </div>
                    <div className="pipe-flange pipe-flange-left"></div>
                    <div className="pipe-flange pipe-flange-right"></div>
                  </div>
                  {/* Pump unit */}
                  <div className={`inline-pump ${pumpActive ? 'pump-running' : ''}`}>
                    <div className="pump-motor">
                      <div className="motor-shaft"></div>
                    </div>
                    <div className="pump-label">PUMP</div>
                  </div>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Footer Stats */}
      <div className="plant-footer">
        <div className="footer-stats">
          <div className="stat-group">
            <span className="stat-label">System Uptime:</span>
            <span className="stat-value">99.7%</span>
          </div>
          <div className="stat-group">
            <span className="stat-label">Water Quality:</span>
            <span className="stat-value">{attackedStage ? '⚠️ WARNING' : '✓ EXCELLENT'}</span>
          </div>
          <div className="stat-group">
            <span className="stat-label">Energy Usage:</span>
            <span className="stat-value">47.2 kW</span>
          </div>
          <div className="stat-group">
            <span className="stat-label">Daily Output:</span>
            <span className="stat-value">357 m³</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlantDiagram;
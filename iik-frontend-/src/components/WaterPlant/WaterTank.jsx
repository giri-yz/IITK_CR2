import React from 'react';
import './WaterTank.css';

const WaterTank = ({ 
  id,
  label,
  capacity = 1000,
  currentLevel = 850,
  sensorId = 'LIT101',
  state = 'normal', // 'normal', 'warning', 'critical', 'attacked', 'contaminated', 'mitigating', 'recovered'
  plcId = 'P101',
  pumpStatus = 'ON',
  pumpFlow = '2.3',
  valveStatus = 'OPEN',
  valvePercent = '75'
}) => {
    console.log(`Tank ${id} - State: ${state}, Level: ${currentLevel}`);
  // Calculate percentage
  const levelPercent = Math.round((currentLevel / capacity) * 100);
  
  // Determine status badge
  const getStatusBadge = () => {
    switch(state) {
      case 'normal':
        return { text: 'Normal', className: 'status-badge--normal' };
      case 'warning':
        return { text: 'Warning', className: 'status-badge--warning' };
      case 'critical':
        return { text: 'Critical', className: 'status-badge--critical' };
      case 'attacked':
        return { text: 'Attacked', className: 'status-badge--attacked' };
      case 'contaminated':
        return { text: 'Contaminated', className: 'status-badge--contaminated' };
      case 'mitigating':
        return { text: 'Mitigating', className: 'status-badge--mitigating' };
      case 'recovered':
        return { text: 'Recovered', className: 'status-badge--recovered' };
      default:
        return { text: 'Normal', className: 'status-badge--normal' };
    }
  };
  
  const statusBadge = getStatusBadge();
  
  return (
    <div className={`water-tank water-tank--${state}`} data-tank-id={id}>
      {/* Header */}
      <div className="tank-header">
        <div className="tank-title">
          <h3>{label}</h3>
          <span className="tank-capacity">Cap: {capacity}mm</span>
        </div>
        <div className="tank-status">
          <span className={`status-badge ${statusBadge.className}`}>
            <span className="status-dot"></span>
            {statusBadge.text}
          </span>
        </div>
      </div>
      
      {/* Body (Tank Visual) */}
      <div className={`tank-body tank-body--${state}`}>
        {/* Sensor Badge */}
        <div className="sensor-badge">
          <span className="sensor-label">{sensorId}</span>
          <span className="sensor-value">{currentLevel}mm</span>
          <span className="sensor-percent">{levelPercent}%</span>
        </div>
        
        {/* Water Fill */}
        <div 
          className={`water-fill water-fill--${state}`} 
          style={{ height: `${levelPercent}%` }}
        >
          <div className="water-surface"></div>
        </div>
        
        {/* Level Text Overlay */}
        <div className="level-indicator">{levelPercent}%</div>
        
        {/* Attack Pulse (hidden by default) */}
        {state === 'attacked' && (
          <div className="attack-pulse-container">
            <div className="attack-pulse"></div>
            <div className="attack-pulse"></div>
            <div className="attack-pulse"></div>
          </div>
        )}
        
        {/* Mitigating Overlay */}
        {state === 'mitigating' && (
          <div className="mitigating-overlay">
            <div className="mitigating-spinner"></div>
            <span className="mitigating-text">AI Processing...</span>
          </div>
        )}
      </div>
      
      {/* Controls */}
      <div className="tank-controls">
        <div className="control-row">
          <span className="control-label">PLC:</span>
          <span className="control-value">{plcId}</span>
          <span className="control-status control-status--active">Active</span>
        </div>
        <div className="control-row">
          <span className="control-label">Pump:</span>
          <span className="control-value">⚡ {pumpStatus}</span>
          <span className="control-status">{pumpFlow} m³/h</span>
        </div>
        <div className="control-row">
          <span className="control-label">Valve:</span>
          <span className="control-value">
            {valveStatus === 'OPEN' ? '🟢' : '🔴'} {valveStatus}
          </span>
          <span className="control-status">{valvePercent}%</span>
        </div>
      </div>
    </div>
  );
};

export default WaterTank;
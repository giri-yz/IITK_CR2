import React, { useEffect, useState } from 'react';
import { checkHealth } from '../../services/api';
import './Header.css';

const Header = ({ systemStatus }) => {
  const [backendStatus, setBackendStatus] = useState('checking');
  
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await checkHealth();
        setBackendStatus(health.status === 'GenTwin backend running' ? 'online' : 'offline');
      } catch (error) {
        setBackendStatus('offline');
      }
    };
    
    checkBackend();
    const interval = setInterval(checkBackend, 5000); // Check every 5s
    
    return () => clearInterval(interval);
  }, []);
  
  const getStatusBadge = () => {
    if (systemStatus === 'ATTACK') {
      return {
        class: 'status-badge--critical',
        text: 'Attack Detected',
        icon: '🚨'
      };
    }
    
    if (backendStatus === 'offline') {
      return {
        class: 'status-badge--warning',
        text: 'Backend Offline',
        icon: '⚠️'
      };
    }
    
    return {
      class: 'status-badge--normal',
      text: 'Normal',
      icon: '✓'
    };
  };
  
  const status = getStatusBadge();
  
  return (
    <header className="dashboard-header">
      <div className="header-left">
        <div className="logo">
          <div className="logo-icon">🌊</div>
          <div className="logo-text">
            <h1>GenTwin</h1>
            <span className="logo-subtitle">Digital Twin Water Treatment</span>
          </div>
        </div>
      </div>
      
      <div className="header-center">
        <div className="live-indicator">
          <span className="live-dot"></span>
          <span className="live-text">LIVE</span>
        </div>
      </div>
      
      <div className="header-right">
        <div className="system-status">
          <span className="status-label">Status:</span>
          <div className={`status-badge ${status.class}`}>
            <span className="status-dot"></span>
            {status.text}
          </div>
        </div>
        
        <div className="system-status">
          <span className="status-label">Backend:</span>
          <div className={`status-badge ${backendStatus === 'online' ? 'status-badge--normal' : 'status-badge--warning'}`}>
            <span className="status-dot"></span>
            {backendStatus === 'online' ? 'Connected' : 'Offline'}
          </div>
        </div>
        
        <button className="icon-button" title="Settings">
          ⚙️
        </button>
        
        <button className="icon-button" title="Help">
          ❓
        </button>
      </div>
    </header>
  );
};

export default Header;
import React from 'react';
import './Placeholder.css';

const Placeholder = ({ title, subtitle, height = '200px' }) => {
  return (
    <div className="panel-placeholder" style={{ minHeight: height }}>
      <h3>{title}</h3>
      {subtitle && <p>{subtitle}</p>}
    </div>
  );
};

export default Placeholder;
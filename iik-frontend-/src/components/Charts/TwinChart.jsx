import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './TwinChart.css';

const TwinChart = ({ attackedStage, attackType }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 700, height: 280 });

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const width = containerRef.current.offsetWidth;
        setDimensions({ width, height: 280 });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Draw chart
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const margin = { top: 20, right: 40, bottom: 40, left: 60 };
    const width = dimensions.width - margin.left - margin.right;
    const height = dimensions.height - margin.top - margin.bottom;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Generate sample data
    const now = Date.now();
    const data = [];
    for (let i = 60; i >= 0; i--) {
      const timestamp = new Date(now - i * 1000);
      const baseValue = 850;
      let realValue = baseValue;
      
      // Simulate attack
      if (attackedStage && i < 45) {
        realValue = baseValue + (45 - i) * 10;
      }
      
      data.push({
        time: timestamp,
        twin: baseValue,
        real: realValue
      });
    }

    // Scales
    const x = d3.scaleTime()
      .domain(d3.extent(data, d => d.time))
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.max(d.twin, d.real)) * 1.1])
      .nice()
      .range([height, 0]);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .selectAll('line')
      .data(y.ticks(5))
      .join('line')
      .attr('x1', 0)
      .attr('x2', width)
      .attr('y1', d => y(d))
      .attr('y2', d => y(d))
      .attr('stroke', 'rgba(255, 255, 255, 0.05)')
      .attr('stroke-dasharray', '2,2');

    // Axes
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x)
        .ticks(6)
        .tickFormat(d3.timeFormat('%H:%M:%S')));

    xAxis.selectAll('text')
      .style('fill', '#6B7280')
      .style('font-size', '11px');

    xAxis.selectAll('path, line')
      .style('stroke', '#374151');

    const yAxis = g.append('g')
      .call(d3.axisLeft(y).ticks(5));

    yAxis.selectAll('text')
      .style('fill', '#6B7280')
      .style('font-size', '11px');

    yAxis.selectAll('path, line')
      .style('stroke', '#374151');

    // Y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -height / 2)
      .attr('text-anchor', 'middle')
      .style('fill', '#9CA3AF')
      .style('font-size', '11px')
      .text('Level (mm)');

    // Check divergence
    const hasDivergence = attackedStage && data.some(d => Math.abs(d.twin - d.real) > 50);

    // Divergence area
    if (hasDivergence) {
      const area = d3.area()
        .x(d => x(d.time))
        .y0(d => y(d.twin))
        .y1(d => y(d.real))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('fill', 'rgba(239, 68, 68, 0.15)')
        .attr('d', area);
    }

    // Twin line (blue)
    const twinLine = d3.line()
      .x(d => x(d.time))
      .y(d => y(d.twin))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#3B82F6')
      .attr('stroke-width', 2.5)
      .attr('d', twinLine);

    // Real line (green or red)
    const realLine = d3.line()
      .x(d => x(d.time))
      .y(d => y(d.real))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', hasDivergence ? '#EF4444' : '#10B981')
      .attr('stroke-width', 2.5)
      .attr('d', realLine);

  }, [dimensions, attackedStage, attackType]);

  const hasDivergence = attackedStage;

  return (
    <div className="twin-chart-container" ref={containerRef}>
      <div className="chart-header">
        <h3>Digital Twin vs Real System</h3>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-line legend-line--twin"></div>
            <span>Digital Twin</span>
          </div>
          <div className="legend-item">
            <div className={`legend-line ${hasDivergence ? 'legend-line--attack' : 'legend-line--normal'}`}></div>
            <span>Real System</span>
          </div>
        </div>
      </div>
      <svg 
        ref={svgRef} 
        width={dimensions.width} 
        height={dimensions.height}
      />
    </div>
  );
};

export default TwinChart;
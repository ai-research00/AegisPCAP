import React, { useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import { Box } from '@mui/material';
import type { NetworkGraph, D3Node, D3Link, NetworkNode } from '../types/network';
import { getRiskColor } from '@utils/formatters';

interface NetworkGraphProps {
  data: NetworkGraph;
  onNodeClick?: (node: NetworkNode) => void;
  width?: number;
  height?: number;
  interactive?: boolean;
}

export const NetworkGraphComponent: React.FC<NetworkGraphProps> = ({
  data,
  onNodeClick,
  width = 1000,
  height = 600,
  interactive = true,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const { nodes: initialNodes, links } = useMemo(() => {
    const nodes: D3Node[] = data.nodes.map((n) => ({ ...n }));
    const links: D3Link[] = data.edges.map((e) => ({
      ...e,
      source: e.source,
      target: e.target,
    }));
    return { nodes: nodes, links };
  }, [data]);

  useEffect(() => {
    if (!svgRef.current || !initialNodes.length) return;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    // Create SVG
    const svg = d3
      .select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height] as any);

    // Create simulation
    const simulation = d3
      .forceSimulation<D3Node>(initialNodes)
      .force(
        'link',
        d3
          .forceLink<D3Node, D3Link>(links)
          .id((d) => d.id)
          .distance(100)
          .strength(0.5)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));

    // Add arrows for directed edges
    svg
      .append('defs')
      .selectAll('marker')
      .data(['arrow'] as string[])
      .enter()
      .append('marker')
      .attr('id', (d: string) => d)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Draw links
    const linkElements = svg
      .append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', (d: D3Link) => {
        const color = getRiskColor(d.riskScore);
        return color;
      })
      .attr('stroke-width', (d: D3Link) => Math.sqrt(d.weight) * 2)
      .attr('opacity', 0.6)
      .attr('marker-end', 'url(#arrow)');

    // Draw nodes
    const nodeElements = svg
      .append('g')
      .selectAll('circle')
      .data(initialNodes)
      .enter()
      .append('circle')
      .attr('r', (d: D3Node) => 15 + Math.sqrt(d.flowCount) * 2)
      .attr('fill', (d: D3Node) => getRiskColor(d.riskScore))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('opacity', 0.85)
      .style('cursor', 'pointer');

    // Add labels
    const labelElements = svg
      .append('g')
      .selectAll('text')
      .data(initialNodes)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('font-size', '11px')
      .attr('fill', '#333')
      .attr('font-weight', 'bold')
      .text((d: D3Node) => {
        const parts = d.id.split('.');
        return parts[parts.length - 1]; // Last octet of IP
      })
      .attr('pointer-events', 'none');

    // Drag behavior
    if (interactive) {
      const drag = d3
        .drag<SVGCircleElement, D3Node>()
        .on('start', (event: any, d: D3Node) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event: any, d: D3Node) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event: any, d: D3Node) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      nodeElements.call(drag as any);
    }

    // Click handler
    nodeElements.on('click', (event: any, d: D3Node) => {
      event.stopPropagation();
      if (onNodeClick) {
        onNodeClick(d);
      }
    });

    // Add tooltip
    nodeElements
      .append('title')
      .text((d: D3Node) => `${d.id}\nRisk: ${d.riskScore.toFixed(1)}\nFlows: ${d.flowCount}`);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      linkElements
        .attr('x1', (d: D3Link) => (typeof d.source === 'object' ? (d.source as D3Node).x || 0 : 0))
        .attr('y1', (d: D3Link) => (typeof d.source === 'object' ? (d.source as D3Node).y || 0 : 0))
        .attr('x2', (d: D3Link) => (typeof d.target === 'object' ? (d.target as D3Node).x || 0 : 0))
        .attr('y2', (d: D3Link) => (typeof d.target === 'object' ? (d.target as D3Node).y || 0 : 0));

      nodeElements.attr('cx', (d: D3Node) => d.x || 0).attr('cy', (d: D3Node) => d.y || 0);

      labelElements.attr('x', (d: D3Node) => d.x || 0).attr('y', (d: D3Node) => d.y || 0);
    });

    return () => {
      simulation.stop();
    };
  }, [initialNodes, links, width, height, interactive, onNodeClick]);

  return <Box component="svg" ref={svgRef} sx={{ border: '1px solid #eee', borderRadius: 1 }} />;
};

export default NetworkGraphComponent;

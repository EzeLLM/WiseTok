import * as d3 from 'd3';

class ForceDirectedGraph {
  constructor(selector, options = {}) {
    this.selector = selector;
    this.width = options.width || 960;
    this.height = options.height || 600;
    this.nodeRadius = options.nodeRadius || 5;
    this.chargeStrength = options.chargeStrength || -30;
    this.linkDistance = options.linkDistance || 30;

    this.svg = null;
    this.simulation = null;
    this.nodes = [];
    this.links = [];
    this.nodeElements = null;
    this.linkElements = null;
    this.labelElements = null;
  }

  init() {
    this.svg = d3.select(this.selector)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .style('border', '1px solid #ccc');

    // Add zoom behavior
    const g = this.svg.append('g');
    this.svg.call(
      d3.zoom()
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        })
    );

    // Create scales
    this.colorScale = d3.scaleOrdinal(d3.schemeCategory10);
    this.sizeScale = d3.scaleSqrt()
      .domain([0, 100])
      .range([3, 15]);

    // Initialize force simulation
    this.simulation = d3.forceSimulation()
      .force('link', d3.forceLink()
        .id(d => d.id)
        .distance(this.linkDistance)
        .strength(0.1))
      .force('charge', d3.forceManyBody()
        .strength(this.chargeStrength))
      .force('center', d3.forceCenter(
        this.width / 2,
        this.height / 2))
      .force('collision', d3.forceCollide()
        .radius(d => this.sizeScale(d.value) + 2));

    this.container = g;
  }

  setData(nodes, links) {
    this.nodes = nodes.map((d, i) => ({
      ...d,
      id: d.id || `node-${i}`,
      value: d.value || 10,
      group: d.group || 0
    }));

    this.links = links.map(d => ({
      ...d,
      source: typeof d.source === 'object' ? d.source.id : d.source,
      target: typeof d.target === 'object' ? d.target.id : d.target,
      strength: d.strength || 1
    }));

    this.update();
  }

  update() {
    // Data join for links
    this.linkElements = this.container.selectAll('line').data(this.links, d => `${d.source}-${d.target}`);

    this.linkElements.exit().remove();

    this.linkElements = this.linkElements.enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.strength) * 2)
      .merge(this.linkElements);

    // Data join for nodes
    this.nodeElements = this.container.selectAll('circle').data(this.nodes, d => d.id);

    this.nodeElements.exit().remove();

    const nodeEnter = this.nodeElements.enter()
      .append('circle')
      .attr('r', d => this.sizeScale(d.value))
      .attr('fill', d => this.colorScale(d.group))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .call(this.drag());

    this.nodeElements = nodeEnter.merge(this.nodeElements);

    // Add tooltips
    this.nodeElements.append('title')
      .text(d => `${d.id}\nValue: ${d.value}`);

    // Data join for labels
    this.labelElements = this.container.selectAll('text').data(this.nodes, d => d.id);

    this.labelElements.exit().remove();

    this.labelElements = this.labelElements.enter()
      .append('text')
      .attr('font-size', 10)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .text(d => d.label || d.id)
      .merge(this.labelElements);

    // Update simulation
    this.simulation.nodes(this.nodes);
    this.simulation.force('link').links(this.links);

    this.simulation.on('tick', () => {
      this.linkElements
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      this.nodeElements
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);

      this.labelElements
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });

    this.simulation.alpha(1).restart();
  }

  drag() {
    return d3.drag()
      .on('start', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  highlightNode(nodeId) {
    this.nodeElements.style('opacity', d =>
      d.id === nodeId || this.isConnected(nodeId, d.id) ? 1 : 0.1
    );

    this.linkElements.style('opacity', d =>
      d.source.id === nodeId || d.target.id === nodeId ? 1 : 0.1
    );
  }

  isConnected(a, b) {
    return this.links.some(d =>
      (d.source.id === a && d.target.id === b) ||
      (d.source.id === b && d.target.id === a)
    );
  }

  clearHighlight() {
    this.nodeElements.style('opacity', 1);
    this.linkElements.style('opacity', 0.6);
  }

  addNodeInteractions() {
    this.nodeElements.on('mouseenter', (event, d) => {
      this.highlightNode(d.id);
    }).on('mouseleave', () => {
      this.clearHighlight();
    });
  }
}

export default ForceDirectedGraph;

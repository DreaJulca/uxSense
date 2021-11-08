var width = 800, height = 200;

var margins = { top: 10, left: 50, right: 50, bottom: 50, between: 20 };

var bottomGraphHeight = 20;
var topGraphHeight = height - (margins.top + margins.bottom + margins.between + bottomGraphHeight);
var graphWidths = width - margins.left - margins.right;

var svg = d3.select('#timeline_hq')
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('font', '10px sans-serif');

svg.append('defs')
    .append('clipPath')
    .attr('id', 'clip')
    .append('rect')
    .attr('width', width)
    .attr('height', height);

var focus = svg
    .append('g')
    .attr('transform', 'translate(' + margins.left + ',' + margins.top + ')');

var context = svg.append('g')
    .attr('class', 'context')
    .attr('transform', 'translate(' + margins.left + ',' +
        (margins.top + topGraphHeight + margins.between) + ')');

var xScaleTop = d3.scaleTime().range([0, graphWidths]),
    xScaleBottom = d3.scaleTime().range([0, graphWidths]),
    yScaleTop = d3.scaleLinear().range([topGraphHeight, 0]),
    yScaleBottom = d3.scaleLinear().range([bottomGraphHeight, 0]);

var xAxisTop = d3.axisBottom(xScaleTop),
    xAxisBottom = d3.axisBottom(xScaleBottom);
// var yAxisTop = d3.axisLeft(yScaleTop);

var lineTop = d3.line()
    .x(function (d) { return xScaleTop(d.date); })
    .y(function (d) { return yScaleTop(d.close); });

var lineBottom = d3.line()
    .x(function (d) { return xScaleBottom(d.date); })
    .y(function (d) { return yScaleBottom(d.close); });

var brush = d3.brushX()
    .extent([[0, 0], [graphWidths, bottomGraphHeight]])
    .on("brush end", function () {
        var s = d3.event.selection || xScaleBottom.range();
        xScaleTop.domain(s.map(xScaleBottom.invert, xScaleBottom));
        focus.select('.x.axis').call(xAxisTop);
        focus.select(".line").attr("d", lineTop);
    });

d3.tsv('https://gist.githubusercontent.com/d3byex/b6b753b6ef178fdb06a2/raw/0c13e82b6b59c3ba195d7f47c33e3fe00cc3f56f/aapl.tsv', function (error, data) {
    data.forEach(function (d) {
        //var formatTime = d3.timeFormat('%d-%b-%y');
        //d.date = formatTime(d.date);
        var parseTime = d3.timeParse('%d-%b-%y');
        d.date = parseTime(d.date);
        d.close = +d.close;
    });

    xScaleTop.domain(d3.extent(data, function (d) { return d.date; }));
    yScaleTop.domain(d3.extent(data, function (d) { return d.close; }));
    xScaleBottom.domain(d3.extent(data, function (d) { return d.date; }));
    yScaleBottom.domain(d3.extent(data, function (d) { return d.close; }));

    var topXAxisNodes = focus.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(' + 0 + ',' + (margins.top + topGraphHeight) + ')')
        .call(xAxisTop);
    styleAxisNodes(topXAxisNodes, 0);

    focus.append('path')
        .datum(data)
        .attr('class', 'line')
        .attr('d', lineTop);

    // var topYAxisNodes = focus.append('g')
    //     .call(yAxisTop);
    // styleAxisNodes(topYAxisNodes);

    context.append('path')
        .datum(data)
        .attr('class', 'line')
        .attr('d', lineBottom);

    var bottomXAxisNodes = context.append('g')
        .attr('transform', 'translate(0,' + bottomGraphHeight + ')')
        .call(xAxisBottom);
    styleAxisNodes(bottomXAxisNodes, 0);

    context.append('g')
        .attr('class', 'x brush')
        .call(brush)
        .selectAll('rect');

    context.selectAll('.selection')
        .attrs({
            stroke: '#000',
            'fill-opacity': 0.125,
            'shape-rendering': 'crispEdges'
        });

    styleLines(svg);
});

function styleLines(svg) {
    svg.selectAll('path.line')
        .attrs({
            fill: 'none',
            'stroke-width': 1.5,
            stroke: 'steelblue',
            'clip-path': 'url(#clip)'
        });
}

function styleAxisNodes(axisNodes, strokeWidth) {
    axisNodes.selectAll('.domain')
        .attrs({
            fill: 'none',
            'stroke-width': strokeWidth,
            stroke: 'black'
        });
    axisNodes.selectAll('.tick line')
        .attrs({
            fill: 'none',
            'stroke-width': 1,
            stroke: 'black'
        });
}
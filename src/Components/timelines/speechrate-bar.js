var video = document.getElementById('video_html5_api');

// set the dimensions and margins of the graph
var margin = { top: 0, right: 50, bottom: 0, left: 50 },
    width = video.width,
    height = 100 - margin.top - margin.bottom;

// set the ranges
var x = d3.scaleBand()
    .range([0, width])
    .padding(0.1);
var y = d3.scaleLinear()
    .range([height, 0]);

// append the svg object to the body of the page
// append a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("speechrate").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

// get the data
d3.csv("modeloutput/TableauUser_Speech_Rate.csv", function (error, data) {
    if (error) throw error;

    // format the data
    data.forEach(function (d) {
        d.sales = +d.sales;
    });

    // Scale the range of the data in the domains
    x.domain(data.map(function (d) { return d.Start; }));
    y.domain([0, d3.max(data, function (d) { return d.Rate; })]);

    // append the rectangles for the bar chart
    svg.selectAll(".bar")
        .data(data)
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function (d) { return x(d.Start); })
        .attr("width", x.bandwidth())
        .attr("y", function (d) { return y(d.Rate); })
        .attr("height", function (d) { return height - y(d.Rate); });

    // add the x Axis
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // add the y Axis
    svg.append("g")
        .call(d3.axisLeft(y));

});
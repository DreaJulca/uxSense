// 2. Use the margin convention practice 
// var margin = { top: 50, right: 50, bottom: 50, left: 50 }
//     , width = window.innerWidth - margin.left - margin.right // Use the window's width 
//     , height = window.innerHeight - margin.top - margin.bottom; // Use the window's height

var margin2 = { top: 10, right: 30, bottom: 30, left: 60 },
    // width = 460 - margin.left - margin.right,
    // height = 400 - margin.top - margin.bottom;
    width2 = 600 - margin2.left - margin2.right,
    height2 = 100 - margin2.top - margin2.bottom;


var svg2 = d3.select("#pitch")
    .append("svg")
    .attr("width", width2 + margin2.left + margin2.right)
    .attr("height", height2 + margin2.top + margin2.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin2.left + "," + margin2.top + ")");


// The number of datapoints
var n = 7553;

// 5. X scale will use the index of our data
var xScale2 = d3.scaleLinear()
    .domain([0, n - 1]) // input
    .range([0, width2]); // output

// 6. Y scale will use the randomly generate number 
var yScale2 = d3.scaleLinear()
    .domain([0, 1]) // input 
    .range([height2, 0]); // output 

// 7. d3's line generator
var line = d3.line()
    .x(function (d, i) { return xScale2(i); }) // set the x values for the line generator
    .y(function (d) { return yScale2(d.y); }) // set the y values for the line generator 
    .curve(d3.curveMonotoneX) // apply smoothing to the line

// 8. An array of objects of length N. Each object has key -> value pair, the key being "y" and the value is a random number
// var dataset = d3.range(n).map(function (d) { return { "y": d3.randomUniform(1)() } })
var dataset = d3.csv('modeloutput/TableauUser_Pitch.csv')

// // 1. Add the SVG to the page and employ #2
// var svg = d3.select("body").append("svg")
//     .attr("width", width + margin.left + margin.right)
//     .attr("height", height + margin.top + margin.bottom)
//     .append("g")
//     .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// 3. Call the x axis in a group tag
svg2.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height2 + ")")
    .call(d3.axisBottom(xScale2)); // Create an axis component with d3.axisBottom

// 4. Call the y axis in a group tag
// svg.append("g")
//     .attr("class", "y axis")
//     .call(d3.axisLeft(yScale)); // Create an axis component with d3.axisLeft

// 9. Append the path, bind the data, and call the line generator 
svg2.append("path")
    .datum(dataset) // 10. Binds data to the line 
    .attr("class", "line") // Assign a class for styling 
    .attr("d", line); // 11. Calls the line generator 

// 12. Appends a circle for each datapoint 
svg2.selectAll(".dot")
    .data(dataset)
    .enter().append("circle") // Uses the enter().append() method
    .attr("class", "dot") // Assign a class for styling
    .attr("cx", function (d, i) { return xScale2(i) })
    .attr("cy", function (d) { return yScale2(d.y) })
    .attr("r", 5);

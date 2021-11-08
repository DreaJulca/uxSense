var video = document.getElementById('video_html5_api');

var margin = { top: 0, right: 50, bottom: 0, left: 50 },
    // width = 460 - margin.left - margin.right,
    // height = 400 - margin.top - margin.bottom;
    height = 100 - margin.top - margin.bottom;
    
//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



// append the svg object to the body of the page
var svg = d3.select("#speech-rate")
    .append("svg")
    .attr("width", "100%")
    .attr("height", (height + margin.top + margin.bottom))
    .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
    .attr("preserveAspectRatio", "none")
    .style('display', 'inline-block')
    .style('position', 'relative')

//    .attr("width", width + margin.left + margin.right)
//    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

//Read the data
d3.csv('modeloutput/TableauUser_Speech_Rate.csv', function (data) {

    var maxEnd = _.max(_.map(data, function(dp){return(1*dp['Start'])}));
    // Add X axis --> it is a date format
    var x = d3.scaleLinear()
        .domain([0, maxEnd])
        .range([0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")");
    // .call(d3.axisBottom(x));

    // Add Y axis
    var y = d3.scaleLinear()
        .domain([0,5.5])
        .range([height, 0]);
    // svg.append("g")
    //     .call(d3.axisLeft(y));

    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function (d) { return d.Start; }).left;

    // Create a rect on top of the svg area: this rectangle recovers mouse position
    svg
        .append('rect')
        .attr('id', 'speechratemouserect')
        .style("fill", "none")
        .style("pointer-events", "all")
        .attr('width', width)
        .attr('height', height)
        .on('mouseover', mouseover)
        .on('mousemove', mousemove)
        .on('mouseout', mouseout)
        .on('click', mouseclick);

    // Add the line
    svg
        .append("path")
        .datum(data)
        .attr("id", "speechratelinepath")
        .attr("maxEnd", maxEnd)
        .attr("origdata", JSON.stringify(data))
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", d3.line()
            .x(function (d) { return x(d.Start) })
            .y(function (d) { return y(d.Rate) })
        )

    //do these after creating the chart path.

    // Create the circle that travels along the curve of chart
    var focus = svg
        .append('g')
        .append('circle')
        .attr('id', 'speechratefocuscircle')
        .style("fill", "none")
        .attr("stroke", "black")
        .attr('r', 8.5)
        .style("opacity", 0)

    // Create bg for the text that travels along the curve of chart
    var focusTextRect = svg
        .append('g')
        .append('rect')
        .attr('id', 'speechratefocustextbg')
        .style("opacity", 0)
        .attr("width", "110px")
        .attr("height", "25px")
        .attr("fill", "lightgrey")

    // Create the text that travels along the curve of chart
    var focusText = svg
        .append('g')
        .append('text')
        .attr('id', 'speechratefocustext')
        .style("opacity", 0)
        .style('text-shadow', '1px 1px 5px white') 
        .attr("text-anchor", "left")
        .attr("alignment-baseline", "middle")

    // What happens when the mouse move -> show the annotations at the right positions.
    function mouseover() {
        focus.style("opacity", 1)
        focusTextRect.style("opacity", 0.75)
        focusText.style("opacity", 1)
    }

    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]
        var minutes = Math.floor((video.duration * selectedData.Start/maxEnd)/60)
        var seconds = Math.round(60 * (((video.duration * selectedData.Start/maxEnd)/60) - minutes))
        var secStr = seconds < 10 ? "0" + seconds.toString() : seconds.toString()
        focus
            .attr("cx", x(selectedData.Start))
            .attr("cy", y(selectedData.Rate))
        focusTextRect
            .attr("x", x(selectedData.Start) + 10)
            .attr("y", y(selectedData.Rate) - 15)
        focusText
            .html("Time:" + minutes.toString() + ":" + secStr + "  -  " + "Rate:" + selectedData.Rate)
            .attr("x", x(selectedData.Start) + 15)
            .attr("y", y(selectedData.Rate))
    }
    function mouseout() {
        focus.style("opacity", 0)
        focusTextRect.style("opacity", 0)
        focusText.style("opacity", 0)
    }

    function mouseclick(){
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]

        video.currentTime = video.duration * selectedData.Start/maxEnd

    }

})
import * as d3 from 'd3';
import React from 'react';

var data = [[0, 50], [100, 80], [200, 40], [300, 60], [400, 30]];

var lineGenerator = d3.line();
var pathString = lineGenerator(data);

d3.select('path')
    .attr('d', pathString);


export default FilterBox => {
    return(<div class="filter-box">
        <svg width="600" height="100">
            <path transform="translate(200, 0)" />
        </svg>
    </div>
    )
}
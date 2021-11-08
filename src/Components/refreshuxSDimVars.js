// set the dimensions and margins of the graph
var aspectRatio = 9/16
var margin = { top: 10, right: 50, bottom: 30, left: 50 }, 
    width = window.innerWidth - (margin.left + margin.right),
    height = aspectRatio * window.innerWidth/7.5 - margin.top - margin.bottom;
    //width = 920 - (margin.left + margin.right),
    //height = 100 - (margin.top + margin.bottom)


var filterheight = 50;
var sliderwidth = 15;
var focusHeight = 10;
var hmargin = 10;
var emotChunkWid = 240;
var annotbtnXPad = 15
var annotbtnYPad = 20



function refreshuxSDimVars(){
    aspectRatio = 9/16
    margin = { top: 10, right: 50, bottom: 30, left: 50 }; 
    width = window.innerWidth - (margin.left + margin.right);
    height = aspectRatio * window.innerWidth/7.5 - margin.top - margin.bottom;
    //for timeline filters
    filterheight = 50;
    sliderwidth = 15;
    //for pandragging
    focusHeight = 10;
    hmargin = 10;
    emotChunkWid = 240;
    //for annot timeline
    annotbtnXPad = 15
    annotbtnYPad = 20


}

export default refreshuxSDimVars;

/*
//marker
var height = 100;
var margin = { top: 0, right: 50, bottom: 0, left: 50 }

//timeline marker again
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 30, left: 50 }


//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



//annotations timeline
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    height = 100 - margin.top - margin.bottom;
    
//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



//action
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    height = 100 - margin.top - margin.bottom;

//windowsize fix
width = window.innerWidth - (margin.left + margin.right)


//emotion
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    height = 100 - margin.top - margin.bottom;
    
//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



//pitch area
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    // width = 460 - margin.left - margin.right,
    // height = 400 - margin.top - margin.bottom;
    height = 70;

//windowsize fix
width = window.innerWidth - (margin.left + margin.right)


//speech rate
var margin = { top: 0, right: 50, bottom: 0, left: 50 },
    // width = 460 - margin.left - margin.right,
    // height = 400 - margin.top - margin.bottom;
    height = 100 - margin.top - margin.bottom;

   //windowsize fix
   width = window.innerWidth - (margin.left + margin.right)



//thumbnails
// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    height = 100 - margin.top - margin.bottom;

//windowsize fix
width = window.innerWidth - (margin.left + margin.right)


//focus zoom
var margin = { top: 10, right: 50, bottom: 10, left: 50 }
var height = 100 - margin.top - margin.bottom;
//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



//pan dragging
var margin = { top: 10, right: 50, bottom: 10, left: 50 }

//windowsize fix
width = window.innerWidth - (margin.left + margin.right)



// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 30, left: 50 }
*/
// var range = document.getElementById('range');

// noUiSlider.create(range, {

//     range: {
//         'min': 1300,
//         'max': 3250
//     },

//     step: 150,

//     // Handles start at ...
//     start: [1450, 2050, 2350, 3000],

//     // ... must be at least 300 apart
//     margin: 300,

//     // ... but no more than 600
//     limit: 600,

//     // Display colored bars between handles
//     connect: true,

//     // Put '0' at the bottom of the slider
//     direction: 'rtl',
//     orientation: 'vertical',

//     // Move handle on tap, bars are draggable
//     behaviour: 'tap-drag',
//     tooltips: true,
//     format: wNumb({
//         decimals: 0
//     }),

//     // Show a scale with the slider
//     pips: {
//         mode: 'steps',
//         stepped: true,
//         density: 4
//     }
// });

document.addEventListener("DOMContentLoaded", function (event) {
    var sppechrate_slider = document.getElementById('speechrate_slider');
    noUiSlider.create(speechrate_slider, {
        start: [20, 80],
        connect: true,
        step: 1,
        range: {
            'min': 0,
            'max': 100
        },
        format: wNumb({
            decimals: 0
        })
    });

    var pitch_slider = document.getElementById('pitch_slider');
    noUiSlider.create(pitch_slider, {
        start: [20, 80],
        connect: true,
        step: 1,
        range: {
            'min': 0,
            'max': 100
        },
        format: wNumb({
            decimals: 0
        })
    });
});
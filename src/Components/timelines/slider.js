// var lowerSlider = document.querySelector('#lower'),
//     upperSlider = document.querySelector('#upper'),
//     lowerVal = parseInt(lowerSlider.value);
// upperVal = parseInt(upperSlider.value);

// upperSlider.oninput = function () {
//     lowerVal = parseInt(lowerSlider.value);
//     upperVal = parseInt(upperSlider.value);

//     if (upperVal < lowerVal + 4) {
//         lowerSlider.value = upperVal - 4;

//         if (lowerVal == lowerSlider.min) {
//             upperSlider.value = 4;
//         }
//     }
// };


// lowerSlider.oninput = function () {
//     lowerVal = parseInt(lowerSlider.value);
//     upperVal = parseInt(upperSlider.value);

//     if (lowerVal > upperVal - 4) {
//         upperSlider.value = lowerVal + 4;

//         if (upperVal == upperSlider.max) {
//             lowerSlider.value = parseInt(upperSlider.max) - 4;
//         }

//     }
// };

$(document).ready(function () {
    $("#slider-range").slider({
        range: true,
        min: 0,
        max: 500,
        values: [75, 300],
        slide: function (event, ui) {
            var value1 = $("#slider-range").slider("values", 0);
            var value2 = $("#slider-range").slider("values", 1);
            $("#slider-range").find(".ui-slider-handle:first").text(value1);
            $("#slider-range").find(".ui-slider-handle:last").text(value2);
        }
    });
});

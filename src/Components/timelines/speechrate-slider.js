document.addEventListener("DOMContentLoaded", function (event) {
    var speechrate_filter = document.getElementById('speechrate-filter');
    noUiSlider.create(speechrate_filter, {
        start: [20, 80],
        connect: true,
        step: 1,
        orientation: "vertical",
        range: {
            'min': 0,
            'max': 100
        },
        format: wNumb({
            decimals: 0
        })
    });

});
var video = document.getElementById("video_html5_api");

var vidTimelineInterval = 30;

function vidTimelineRecur(i, duration){

    if(i < duration){
        video.currentTime = i;
        generateThumbnail(i);
        setTimeout(vidTimelineRecur(i+vidTimelineInterval, duration), 500)
    } else {
        video.currentTime = 0;
    }
}
video.addEventListener('loadeddata', function () {
        var duration = video.duration;
        vidTimelineRecur(0, duration)
    });

function generateThumbnail(i) {
    var canvas = document.getElementById('myCanvas');
    var context = canvas.getContext('2d');
    context.drawImage(video, (i/vidTimelineInterval) * 12.5, 0, 100, 50);
    var dataURL = canvas.toDataURL();
    var img = document.createElement('img');
    img.setAttribute('src', dataURL);                                                    
    document.getElementById('thumbnails').appendChild(img);
}

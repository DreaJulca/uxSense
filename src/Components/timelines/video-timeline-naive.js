var video = document.getElementById("video_html5_api");
var vidDuration = 0;
var vidTimelineInterval = 30;
var vidTimelineFrameCount = 10;


function vidTimelineRecur(i){
    if(i < vidDuration){
        setTimeout('generateThumbnail(' + i + ')', 500);
    }
}


video.addEventListener('loadeddata', function () {
        video.play();
        vidDuration = video.duration;
        vidTimelineInterval = Math.round(vidDuration/vidTimelineFrameCount);
        vidTimelineRecur(0)
    }, once=true);

    function generateThumbnail(i) {
        var canvas = document.getElementById('myCanvas');
        var context = canvas.getContext('2d');
        //context.drawImage(video, (i/vidTimelineInterval) * 100, 0, 100, 80);
        //var dataURL = canvas.toDataURL();
        var thumbnail = new Image();
        thumbnail.src = 'frames/frame' + i.toString() + '.png';
        thumbnail.onload = function(){
            context.drawImage(thumbnail, (i/vidTimelineInterval) * 50, 0, 100, 80);
            vidTimelineRecur(i+vidTimelineInterval);
        }
    }


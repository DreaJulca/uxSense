var video = document.getElementByID('video_html5_api');
var vidDur = video.duration;

var thumbCount = 8;

var thumbTimes = [];
for (i = 0; i < thumbCount; i++) {
    thumbTimes.append[Math.round(vidDur / (thumbCount - i))];
}
thumbTimes.reverse();

var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');
for (i = 0; i < thumbCount; i++) {
    //set the video time
    video.currentTime = thumbTimes[i];

    //get the frame and render to canvas; offset dx for each thumbnail
    context.drawImage(video, i * 50, 0, 50, 50);
}

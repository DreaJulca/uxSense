var sliderImgCount = 11;
var video = document.getElementById('video_html5_api');
var canvas = document.getElementById('myCanvas');
var context = canvas.getContext('2d');
//context.drawImage(video, (i/vidTimelineInterval) * 100, 0, 100, 80);
//var dataURL = canvas.toDataURL();
//Add some initial events
var thumbnail = new Image();
thumbnail.src = 'frames/_initCanvas.png';
thumbnail.onload = function(){
    context.drawImage(thumbnail, 0, 0, 600, 80);

    var thumbnails = document.getElementById('thumbnails');
    for(i=0; i < (Math.floor(video.duration)+1); i++){
        var img = document.createElement('img');
        img.src = 'frames/frame'+i.toString()+'.png';
        thumbnails.appendChild(img);
    }
}

//Now add some video playback and mouse event triggers to canvas
function thumbnailmouseover(event){
    var mousex = event.PageX;
    var mousey = event.PageY;

    

}

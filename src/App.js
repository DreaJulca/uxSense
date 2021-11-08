import './App.css';
import './Components/timelines/main.js';
import videojs from 'videojs';


var video = document.getElementById("video_html5_api");

var vidTimelineInterval = 30;

function vidTimelineRecur(i, duration){

    if(i < duration){
        video.currentTime = i;
        generateThumbnail(i);
        setTimeout(vidTimelineRecur(i+vidTimelineInterval, duration), 500)
    }
    else return;
}
function generateThumbnail(i) {
        var canvas = document.getElementById('myCanvas');
        var context = canvas.getContext('2d');
        // context.drawImage(video, (i / vidTimelineInterval) * 10, 0, 100, 50);
        context.drawImage(video, i, 0, 100, 50);
        var dataURL = canvas.toDataURL();

        var img = document.createElement('img');
        img.setAttribute('src', dataURL);

        document.getElementById('thumbnails').appendChild(img);
}
video.addEventListener('loadeddata', function () {
        var duration = video.duration;
        vidTimelineRecur(0, duration)
        video.currentTime = 1;
});

function App() {
  return (
    <div className="App">
      <header className="App-header">
      </header>
      <div class="grid-container">
        <div class="left">
            <div class="menu">
                <div class="menu-box">
                    <nav style="background-color: darkolivegreen">
                        <div class="nav-wrapper">
                            <a href="#" class="brand-logo center" style="color:white">
                                <bold>uxSense</bold>
                            </a>
                        </div>
                    </nav>
                </div>
            </div>
            <div class="video">
                <div class="video-box">
                    <video id="video_html5_api" width="600" height="350" style="text-align: center;" controls>
                        <source src="1-p6-coffeemachine.mp4" type="video/mp4">
                        <source src="1-p6-coffeemachine.ogg" type="video/ogg"> Your browser does not support HTML5 video.
                    </video>
                </div>
            </div>
            <div class="timelines">
                <div class="timelines-box">
                    <div class="responsive-table table-status-sheet">
                        <table class="highlight bordered" style="height:310px;">
                            <thead>
                                <tr>
                                    <th class="center" style="width: 20%;">Features</th>
                                    <th class="center" style="width: 80%;">Timelines</th>
                                </tr>
                            </thead>
                            <tbody style="height:250px;">
                                <tr>
                                    <td style="width: 20%;">Video</td>
                                    <td style="width: 80%;">

                                        <canvas id="myCanvas" width="600" height="100" style="border:1px solid #d3d3d3;">
                                           <!-- <script type="text/javascript" src="video-timeline.js"></script> -->
                                           <div id="thumbnails"></div>
                                        </canvas>
                                        <script>
                                        </script>

                                        <!-- <script type="text/javascript" src="video-timeline.js"></script> -->
                                    </td>
                                </tr>
                                <tr>
                                    <td style="width: 20%;">Speech Rate</td>
                                    <td style="width: 80%;">
                                        <div id="speech-rate"></div>
                                        <script type="text/javascript" src="basic-line.js"></script>
                                        <!-- <script type="text/javascript" src="linechart.js"></script> -->
                                    </td>
                                </tr>
                                <tr>
                                    <td style="width: 20%;">Pitch</td>
                                    <td style="width: 80%;">
                                        <div id="pitch"></div>
                                        <script type="text/javascript" src="linechart.js"></script>
                                    </td>
                                </tr>
                                <tr>
                                    <td>Action1</td>
                                </tr>
                                <tr>
                                    <td>Action2</td>
                                </tr>
                                <tr>
                                    <td>Posture</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        <div class="right">
            <div class="search">
                <div class="search-bar">
                    <!-- <h6>Search and Filter</h6> -->
                    <nav>
                        <div class="nav-wrapper">
                            <form>
                                <div class="input-field">
                                    <input id="search" type="search" required placeholder="Search and Filter">
                                    <label class="label-icon" for="search">
                                        <i class="material-icons" style="color:gray">search</i>
                                    </label>
                                    <i class="material-icons" style="color:gray">close</i>
                                </div>
                            </form>
                        </div>
                    </nav>
                </div>
            </div>
            <div class="filter">
                <div class="filter-box">
                    <svg width="600" height="100">
                        <path transform="translate(200, 0)" />
                    </svg>
                    <script>
                        var data = [[0, 50], [100, 80], [200, 40], [300, 60], [400, 30]];

                        var lineGenerator = d3.line();
                        var pathString = lineGenerator(data);

                        d3.select('path')
                            .attr('d', pathString);

                    </script>
                </div>
            </div>
            <div class="annotations">
                <div class="annotations-box">
                    <form accept-charset="UTF-8" action="action_page.php" autocomplete="off" method="GET" target="_blank">
                        <fieldset>
                            <legend> Annotation </legend>
                            <label for="name">Notes</label>
                            <textarea style="height:100px;"></textarea>
                            <br/>
                            <button class="btn" type="submit" value="Submit">Submit</button>
                        </fieldset>
                    </form>
                </div>
            </div>
            <div class="tags">
                <div class="tags-table">
                    <div class="responsive-table table-status-sheet">
                        <table class="highlight bordered">
                            <!-- <colgroup>
                                <col span="1" style="width: 20%;">
                                <col span="1" style="width: 30%;">
                                <col span="1" style="width: 50%;">
                            </colgroup> -->
                            <thead>
                                <tr>
                                    <th class="center" style="width: 20%;">Timestamp</th>
                                    <th class="center" style="width: 30%;">Tags</th>
                                    <th class="center" style="width: 50%;">Annotations</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="width: 20%;">1</td>
                                    <td style="width: 30%;">1</td>
                                    <td style="width: 50%;">Note1</td>
                                </tr>
                                <tr>
                                    <td>2</td>
                                </tr>
                                <tr>
                                    <td>3</td>
                                </tr>
                                <tr>
                                    <td>4</td>
                                </tr>
                                <tr>
                                    <td>5</td>
                                </tr>
                                <tr>
                                    <td>6</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    </div>
  );
}

export default App;

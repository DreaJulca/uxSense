// The timelines don't actually exist as components yet; may need to start from scratch for some of this stuff.
//import ActionTimeline from './actions';
import React from 'react';

export default function Timelines( props ) {
    return(<div class="responsive-table table-status-sheet">
        <table class="highlight bordered" style={{height:"310px"}}>
            <thead>
                <tr>
                    <th class="center" style={{width: "20%"}}>Features</th>
                    <th class="center" style={{width: "80%"}}>Timelines</th>
                </tr>
            </thead>
            <tbody style={{height:"250px;"}}>
                <tr>
                    <td style={{width: "20%"}}>Video</td>
                    <td style={{width: "80%"}}>

                        <canvas id="myCanvas" width="600" height="100" style={{border:"1px solid #d3d3d3;"}}>
                        <div id="thumbnails"></div>
                        </canvas>
                        <script>
                        </script>
                    </td>
                </tr>
                <tr>
                    <td style={{width: "20%"}}>Speech Rate</td>
                    <td style={{width: "80%"}}>
                        <div id="speech-rate"></div>
                        <script type="text/javascript" src="basic-line.js"></script>
                    </td>
                </tr>
                <tr>
                    <td style={{width: "20%"}}>Pitch</td>
                    <td style={{width: "80%"}}>
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
    )
}
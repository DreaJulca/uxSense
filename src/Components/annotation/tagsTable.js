import React from 'react';

export default TagsTable => {
    
return (<div class="tags-table">
<div class="responsive-table table-status-sheet">
    <table class="highlight bordered">
            <col span="1" style={{width: "20%"}} />
            <col span="1" style={{width: "30%"}} />
            <col span="1" style={{width: "50%"}} />
        <thead>
            <tr>
                <th class="center" style={{width: "20%"}}>Timestamp</th>
                <th class="center" style={{width: "30%"}}>Tags</th>
                <th class="center" style={{width: "50%"}}>Annotations</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style={{width: "20%"}}>1</td>
                <td style={{width: "30%"}}>1</td>
                <td style={{width: "50%"}}>Note1</td>
            </tr>
        </tbody>
    </table>
</div>
</div>
)
}

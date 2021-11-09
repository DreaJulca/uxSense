import React from 'react';

export default TagsTable => {
    
return (<div className="tags-table">
    <table className="highlight bordered">
        <thead>
            <tr>
                <th className="center" style={{width: "20%"}}>Timestamp</th>
                <th className="center" style={{width: "30%"}}>Tags</th>
                <th className="center" style={{width: "50%"}}>Annotations</th>
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
)
}

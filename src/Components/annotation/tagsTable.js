import React from 'react';

export default TagsTable => {
    
return (<div className="tags-table" style={{height: "100%", width: "100%"}}>
    <table className="highlight bordered" style={{height: "100%", width: "100%"}}>
        <thead style={{height: "5%", width: "100%"}}>
            <tr>
                <th className="center" style={{width: "20%"}}>Timestamp</th>
                <th className="center" style={{width: "30%"}}>Tags</th>
                <th className="center" style={{width: "50%"}}>Annotations</th>
            </tr>
        </thead>
        <tbody style={{height: "95%", width: "100%"}}>
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

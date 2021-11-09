import React from 'react';

export default AnnotationsBox => {
    return(<div className="annotations-box">
        <form acceptCharset="UTF-8" action="action_page.php" autoComplete="off" method="GET" target="_blank">
            <fieldset>
                <legend> Annotation </legend>
                <label htmlFor="name">Notes</label>
                <textarea style={{height:'100px'}}></textarea>
                <br/>
                <button className="btn" type="submit" value="Submit">Submit</button>
            </fieldset>
        </form>
    </div>
    )
}
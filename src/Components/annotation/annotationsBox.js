export default AnnotationsBox => {
    return(<div class="annotations-box">
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
    )
}
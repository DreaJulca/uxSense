import React from 'react';

export default SearchBar => {
return (<div className="search-bar">
    <nav>
        <div className="nav-wrapper">
            <form>
                <div className="input-field">
                    <input id="search" type="search" required placeholder="Search and Filter" />
                    <label className="label-icon" htmlFor="search">
                        <i className="material-icons" style={{color:"gray"}}>search</i>
                    </label>
                    <i className="material-icons" style={{color:"gray"}}>close</i>
                </div>
            </form>
        </div>
    </nav>
    </div>
)}
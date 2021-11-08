import 'materialize-css/dist/css/materialize.css'
import 'materialize-css'

document.addEventListener('DOMContentLoaded', function () {
    //   M.AutoInit();
    var elem = document.querySelector('.collapsible.expandable');
    var instance = M.Collapsible.init(elem, {
        accordion: false
    });
});
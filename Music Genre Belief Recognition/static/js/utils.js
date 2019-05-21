'use strict';

function lowerBound(array, element, key) {
    key = key || function(x) { return x; };

    var begin = 0;
    var end = array.length;
    while(begin < end) {
        var m = floor((begin + end) / 2);
        if(key(array[m]) >= element)
            end = m;
        else
            begin = m + 1;
    }
    return begin;
}

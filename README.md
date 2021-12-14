# Movement
Video Movement Analysis for Matlab

Single camera video analysis done by frame by frame calculation

More information in comments of Movement.m



For those running into the following error:

Array formation and parentheses-style indexing with objects of class 'VideoReader' is not allowed.  Use objects of class 'VideoReader' only as scalars or use a cell array.

Error in Movement (line 131)
        vidObj(i) = VideoReader(fname{i});

use Movement_v2 instead
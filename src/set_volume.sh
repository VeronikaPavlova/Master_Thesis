#!/usr/bin/env bash
amixer -D hw:USB sset 'Master',0 80% unmute
amixer -D hw:USB sset 'Line',0 0% mute cap capture 80%
amixer -D hw:USB sset 'Line',1 0% mute

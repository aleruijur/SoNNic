--[[
Demo.lua

This script plays selected tracks forever in a loop, for purposes of demoing the AI. ]]--

--[[ BEGIN CONFIGURATION ]]--
TRACKS = {'../states/MuteCityTA.state', '../states/MuteCityGP2.state', '../states/MuteCity2.state', '../states/MuteCityGP.state'}
PLAY_FOR_FRAMES = {2300,500,500,800} -- Number of frames that each track is going to play
--[[ END CONFIGURATION ]]--

local util = require("util")
local play = loadfile("Play.lua")

client.unpause()
event.onexit(function()
  client.pause()
end)

local state = 1
local frame = 0
while true do
    -- Load the savestate from the list
    print("Loading:", TRACKS[state])
    savestate.load(TRACKS[state])
    play(PLAY_FOR_FRAMES[state]) -- Play for just some frames on this track
    state = state + 1    -- Go to the next track.
    if state > #TRACKS then state = 1 end
end

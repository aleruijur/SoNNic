--[[ BEGIN CONFIGURATION ]]--
RECORD_EVERY = 1 -- 10 Record input and make a screenshot every # of frames
PLAY_FOR = 2000 -- 100 Target input and screenshots on this session. Game will end once goal is reached
USE_MAPPING = true
--[[ END CONFIGURATION ]]--

local util = require("util")

-- Generate a recording id.
local RECORDING_ID = util.generateUUID()
print("Recording ID:", RECORDING_ID)
-- Ensure that there is a recordings folder, as well as a subfolder for the current track-mode combination.
os.execute('mkdir ..\\recordings\\')
-- Create a folder for this recording.
RECORDING_FOLDER = '..\\recordings\\' .. '\\record-' .. RECORDING_ID
os.execute('mkdir ' .. RECORDING_FOLDER)
-- Create an empty inputs file that will be appended to.
os.execute('type nul > ' .. RECORDING_FOLDER .. '\\inputs.txt')

client.unpause()
client.speedmode(50)

function onexit()
  if inputs_file ~= nil then
    inputs_file:close()
  end
  client.pause()
  client.speedmode(100)
end

local exit_guid = event.onexit(onexit)

local recording_frame = 1

local inputs_file = io.open(RECORDING_FOLDER .. '\\inputs.txt', 'a')
local action = {}
local start = false
while recording_frame < PLAY_FOR do
    action = joypad.get()
    -- It starts recording when the player starts walking avoiding the initial static recordings
    if not start and action["P1 Right"] then start = true end

    if start then 
      client.screenshot(RECORDING_FOLDER .. '\\' .. recording_frame .. '.png')
      inputs_file:write(util.addInputLine(action) .. '\n')
      inputs_file:flush()
      recording_frame = recording_frame + 1
    end

    for i=0, RECORD_EVERY do emu.frameadvance() end
end
onexit()
event.unregisterbyid(exit_guid)

return recording_frame
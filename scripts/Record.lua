--[[ BEGIN CONFIGURATION ]]--
RECORD_EVERY = 10 -- Record imput and make a screenshot every # of frames
PLAY_FOR = 550 -- Target imput and screenshots on this session. Game will end once goal is reached
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
-- Create an empty steering file that will be appended to.
os.execute('type nul > ' .. RECORDING_FOLDER .. '\\steering.txt')

client.unpause()
client.speedmode(75)

function onexit()
  if steering_file ~= nil then
    steering_file:close()
  end
  client.pause()
  client.speedmode(100)
end

local exit_guid = event.onexit(onexit)

local recording_frame = 1
if RECORDING_START_FRAME ~= nil then recording_frame = RECORDING_START_FRAME end

local steering_file = io.open(RECORDING_FOLDER .. '\\steering.txt', 'a')
local action={}
while recording_frame < PLAY_FOR do
    action=joypad.get()
    client.screenshot(RECORDING_FOLDER .. '\\' .. recording_frame .. '.png')
    steering_file:write(action["P1 X Axis"] .. '\n')
    steering_file:flush()
    recording_frame = recording_frame + 1
    for i=0, RECORD_EVERY do emu.frameadvance() end

end
onexit()
event.unregisterbyid(exit_guid)

return recording_frame

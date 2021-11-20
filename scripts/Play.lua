--[[
Play.lua

This script is used to actually play the Convolutional AI on a track. It asynchronously communicates
with predict-server.py over a TCP socket. The message protocol is a simple, line-oriented feed.
This module can also be called as a function, in which case the first argument is the number of
frames to play for.
]]--

--[[ BEGIN CONFIGURATION ]]--
USE_CLIPBOARD = false -- Use the clipboard to send screenshots to the predict server.

--[[ How many frames to wait before sending a new prediction request. If you're using a file, you
may want to consider adding some frames here. ]]--
WAIT_FRAMES = 1

USE_MAPPING = true -- Whether or not to use input remapping.

--[[ END CONFIGURATION ]]--

local chunk_args = {...}
local PLAY_FOR_FRAMES = chunk_args[1]
print("Connecting to predict server")

local util = require("util")

local SCREENSHOT_FILE = util.getTMPDir() .. '\\predict-screenshot.png'

local tcp = require("lualibs.socket").tcp()
local success, error = tcp:connect('localhost', 36296)
if not success then
  print("Failed to connect to server:", error)
  print("REMEMBER: You must run predict-server.py first!")
  return
end

client.setscreenshotosd(false)

tcp:send("START" .. "\n")

tcp:settimeout(0)

client.unpause()

outgoing_message, outgoing_message_index = nil, nil
function request_prediction()
  if USE_CLIPBOARD then
    client.screenshottoclipboard()
    outgoing_message = "PREDICTFROMCLIPBOARD\n"
  else
    client.screenshot(SCREENSHOT_FILE)
    outgoing_message = "PREDICT:" .. SCREENSHOT_FILE .. "\n"
  end
  outgoing_message_index = 1
end
request_prediction()

local receive_buffer = ""

function onexit()
  client.pause()
  tcp:close()
end
local exit_guid = event.onexit(onexit)

local current_action = 0
local frame = 1
local esc_prev = input.get()['Escape']

BOX_CENTER_X, BOX_CENTER_Y = 160, 215
BOX_WIDTH, BOX_HEIGHT = 100, 4
SLIDER_WIDTH, SLIDER_HIEGHT = 4, 16
function draw_info()
  gui.drawBox(BOX_CENTER_X - BOX_WIDTH / 2, BOX_CENTER_Y - BOX_HEIGHT / 2,
              BOX_CENTER_X + BOX_WIDTH / 2, BOX_CENTER_Y + BOX_HEIGHT / 2,
              none, 0x60FFFFFF)
  gui.drawBox(BOX_CENTER_X + current_action*(BOX_WIDTH / 2) - SLIDER_WIDTH / 2, BOX_CENTER_Y - SLIDER_HIEGHT / 2,
              BOX_CENTER_X + current_action*(BOX_WIDTH / 2) + SLIDER_WIDTH / 2, BOX_CENTER_Y + SLIDER_HIEGHT / 2,
              none, 0xFF00FF00)
end

while true do

  -- Process the outgoing message.
  if outgoing_message ~= nil then
    local sent, error, last_byte = tcp:send(outgoing_message, outgoing_message_index)
    if sent ~= nil then
      outgoing_message = nil
      outgoing_message_index = nil
    else
      if error == "timeout" then
        outgoing_message_index = last_byte + 1
      else
        print("Send failed: ", error); break
      end
    end
  end

  local message, error
  message, error, receive_buffer = tcp:receive("*l", receive_buffer)
  if message == nil then
    if error ~= "timeout" then
      print("Receive failed: ", error); break
    end
  else
    if message ~= "PREDICTIONERROR" then
      current_action = tonumber(message)
      for i=1, WAIT_FRAMES do
        joypad.set({["P1 A"] = true})
        joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action, USE_MAPPING)})
        draw_info()
        emu.frameadvance()
      end
    else
      print("Prediction error...")
    end
    request_prediction()
  end
  joypad.set({["P1 A"] = true})
  if current_action > 0.95 or current_action < -0.95 then
    joypad.setanalog({["P1 X Axis"] = current_action*18})
  else
    joypad.setanalog({["P1 X Axis"] = current_action*32})
  end
  if WAIT_FRAMES < 1 then
    joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action, USE_MAPPING)})
  end
  draw_info()
  emu.frameadvance()

  if PLAY_FOR_FRAMES ~= nil then
    if PLAY_FOR_FRAMES > 0 then PLAY_FOR_FRAMES = PLAY_FOR_FRAMES - 1
    elseif PLAY_FOR_FRAMES == 0 then break end
  end
  frame = frame + 1

  if not esc_prev and input.get()['Escape'] then break end
  esc_prev = input.get()['Escape']
end

onexit()
event.unregisterbyid(exit_guid)

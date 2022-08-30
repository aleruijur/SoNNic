local util = {}

local uuid = require("lualibs.uuid"); uuid.seed()
function util.generateUUID()
  return uuid()
end

-- Return the location of the TMP dir on this computer, caching the result.
local TMP_DIR = nil
function util.getTMPDir()
  if TMP_DIR == nil then TMP_DIR = io.popen("echo %TEMP%"):read("*l") end
  return TMP_DIR
end

return util
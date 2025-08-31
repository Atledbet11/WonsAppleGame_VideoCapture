#pragma once
#include <string>
#include <iostream>

namespace utils {

// Runs a shell command and returns stdout as a string.
// Windows: uses _popen; POSIX: popen.
std::string runCommand(const std::string& cmd);

} // namespace utils

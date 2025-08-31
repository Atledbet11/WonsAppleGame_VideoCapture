#include "utils.hpp"

#ifdef _WIN32
#include <cstdio>
#include <windows.h>
#else
#include <cstdio>
#endif

#include <filesystem>

namespace utils {

	// Portable-ish command runner (Windows: _popen)
	std::string runCommand(const std::string& cmd) {
		std::string data;

	#ifdef _WIN32
		// Redirect STDERR into STDOUT for the command.
		std::string full = cmd + " 2>&1";

		FILE* pipe = _popen(full.c_str(), "r");

		if (!pipe) {
			std::cerr << "Unable to open command pipe.\n";
			return data;
		}

		char buffer[4096];
		while (fgets(buffer, sizeof(buffer), pipe)) {
			data += buffer;
		}

		_pclose(pipe);
	#else
		// POSIX popen fallback if you later want it on Linux
		FILE* pipe = popen(cmd.c_str(), "r");
		if (!pipe) return data;
		char buffer[4096];
		while (fgets(buffer, sizeof(buffer), pipe)) {
			data.append(buffer);
		}
		pclose(pipe);
	#endif
		return data;
	}

} // namespace utils

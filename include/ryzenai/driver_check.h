#pragma once

#include <string>

namespace ryzenai {

// Checks the NPU driver version and warns if it's too old
// Returns true if the driver check passes, false otherwise
bool CheckNPUDriverVersion();

// Returns the NPU architecture ("Medusa (XDNA3)", "XDNA2") or "" if undetected
std::string GetNPUArchitecture();

}


#pragma once

#include <string>
#include <string_view>



namespace Noether
{

// Lowercase hex SHA-256 (FIPS 180-4) of `data`; identifies sources in run results and ledgers.
std::string sha256Hex(std::string_view data);

} // namespace Noether

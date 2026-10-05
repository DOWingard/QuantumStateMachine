#include "Sha256.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>



namespace Noether
{

std::string sha256Hex(std::string_view data)
{
    static constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::vector<unsigned char> msg(data.begin(), data.end());
    const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int s = 56; s >= 0; s -= 8) msg.push_back(static_cast<unsigned char>(bits >> s));

    for (std::size_t off = 0; off < msg.size(); off += 64)
    {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t t = 0; t < 16; ++t)
            w[t] = (std::uint32_t{msg[off + 4 * t]} << 24) | (std::uint32_t{msg[off + 4 * t + 1]} << 16) |
                   (std::uint32_t{msg[off + 4 * t + 2]} << 8) | std::uint32_t{msg[off + 4 * t + 3]};
        for (std::size_t t = 16; t < 64; ++t)
        {
            const std::uint32_t s0 = std::rotr(w[t - 15], 7) ^ std::rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            const std::uint32_t s1 = std::rotr(w[t - 2], 17) ^ std::rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        std::array<std::uint32_t, 8> v = h;
        for (std::size_t t = 0; t < 64; ++t)
        {
            const std::uint32_t S1 = std::rotr(v[4], 6) ^ std::rotr(v[4], 11) ^ std::rotr(v[4], 25);
            const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const std::uint32_t t1 = v[7] + S1 + ch + k[t] + w[t];
            const std::uint32_t S0 = std::rotr(v[0], 2) ^ std::rotr(v[0], 13) ^ std::rotr(v[0], 22);
            const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            const std::uint32_t t2 = S0 + maj;
            v = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
        }
        for (std::size_t t = 0; t < 8; ++t) h[t] += v[t];
    }

    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const std::uint32_t x : h)
        for (int s = 28; s >= 0; s -= 4) out.push_back(hex[(x >> s) & 0xF]);
    return out;
}

} // namespace Noether

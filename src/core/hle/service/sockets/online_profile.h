// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Service::Sockets::OnlineProfile {

struct Profile {
    std::string server;
    std::string code;
    std::string secret;
    std::vector<std::string> friends;
    std::vector<std::string> removed;
};

std::filesystem::path ProfilesDir();
std::optional<Profile> LoadProfile(const std::string& server, bool create);
std::string ProfileText(const Profile& profile);
std::optional<Profile> ParseProfile(std::string_view text);
std::string NewSecret();
std::vector<std::string> ParseCodes(std::string_view text);
std::string IdentityToken(std::string_view build);
void AddFriend(Profile& profile, const std::string& code);

} // namespace Service::Sockets::OnlineProfile

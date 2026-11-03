// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright Ahmad Ali Parr
// Tiny --key=value argument parser shared by the apps and benches.
#pragma once
#include <cstdlib>
#include <cstdio>
#include <map>
#include <string>

namespace skapp {

struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        kv["_pos" + std::to_string(kv.size())] = a;
        continue;
      }
      auto eq = a.find('=');
      if (eq == std::string::npos) kv[a.substr(2)] = "1";
      else kv[a.substr(2, eq - 2)] = a.substr(eq + 1);
    }
  }
  bool has(const std::string& k) const { return kv.count(k) != 0; }
  std::string str(const std::string& k, const std::string& d) const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  int i(const std::string& k, int d) const { return has(k) ? std::atoi(kv.at(k).c_str()) : d; }
  double f(const std::string& k, double d) const { return has(k) ? std::atof(kv.at(k).c_str()) : d; }
};

}  // namespace skapp

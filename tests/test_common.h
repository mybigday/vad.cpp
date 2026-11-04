#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static int g_failures = 0;

// test data location: compiled-in default, overridable (e.g. when running on another device)
static std::string env_or(const char * name, const char * def) {
    const char * v = getenv(name);
    return v && *v ? v : def;
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_failures; } } while (0)

static bool file_exists(const std::string & p) { return (bool) std::ifstream(p); }

// exit code 77 = skipped (missing test data)
static void require_file(const std::string & p) {
    if (!file_exists(p)) { fprintf(stderr, "SKIP: %s missing (run scripts/prepare-test-data.sh)\n", p.c_str()); exit(77); }
}

static int finish(const char * name) {
    if (g_failures) { fprintf(stderr, "%s: %d failure(s)\n", name, g_failures); return 1; }
    fprintf(stderr, "%s: ok\n", name);
    return 0;
}

#include "commands.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <inspect|run|chat|bench|vae-decode> [args...]\n", argv[0]);
        return 1;
    }

    std::string cmd = argv[1];
    // shift argv so the subcommand handler sees argv[0]=<subcommand name>, argv[1..]=its own args
    if (cmd == "inspect") return cmd_inspect(argc - 1, argv + 1);
    if (cmd == "run")     return cmd_run(argc - 1, argv + 1);
    if (cmd == "chat")    return cmd_chat(argc - 1, argv + 1);
    if (cmd == "bench")      return cmd_bench(argc - 1, argv + 1);
    if (cmd == "vae-decode") return cmd_vae_decode(argc - 1, argv + 1);

    fprintf(stderr, "unknown subcommand '%s' (expected inspect|run|chat|bench|vae-decode)\n", cmd.c_str());
    return 1;
}

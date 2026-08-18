#include "clip_tokenizer.h"
#include "clip_merges_data.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <regex>
#include <sstream>

// Same bytes_to_unicode() trick as model.cpp's GPT-2 tokenizer (see its build_byte_tables), kept
// as an independent copy here since CLIP additionally needs the *iteration order* of the "bs" list
// (not just the byte->string map) to assign vocab ids 0..511 in the exact order the standard CLIP
// vocab.json uses -- ids must line up with the SDXL checkpoint's token_embedding rows, so this has
// to match the reference bytes_to_unicode() bit-for-bit, not just be "an equivalent" mapping.
static void build_byte_tables(std::array<uint8_t, 256> & bs_order, std::array<std::string, 256> & byte_encoder) {
    std::vector<int> bs;
    for (int b = 33;  b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        bs_order[i] = (uint8_t) bs[i];
        uint32_t cp = (uint32_t) cs[i];
        std::string enc;
        if (cp < 0x80) {
            enc.push_back((char) cp);
        } else if (cp < 0x800) {
            enc.push_back((char)(0xC0 | (cp >> 6)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            enc.push_back((char)(0xE0 | (cp >> 12)));
            enc.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        }
        byte_encoder[(uint8_t) bs[i]] = enc;
    }
}

// merges byte-encoded symbols for one word until no adjacent pair has a known rank -- identical
// algorithm to model.cpp's bpe_merge, kept as its own copy since this tokenizer is standalone.
static std::vector<std::string> bpe_merge(std::vector<std::string> symbols,
                                           const std::unordered_map<std::string, int> & ranks) {
    while (symbols.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            auto it = ranks.find(symbols[i] + "\x01" + symbols[i + 1]);
            if (it != ranks.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank < 0) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

static std::string whitespace_clean_lower(const std::string & text) {
    std::string collapsed = std::regex_replace(text, std::regex(R"(\s+)"), " ");
    size_t start = collapsed.find_first_not_of(' ');
    size_t end   = collapsed.find_last_not_of(' ');
    std::string stripped = (start == std::string::npos) ? "" : collapsed.substr(start, end - start + 1);
    std::transform(stripped.begin(), stripped.end(), stripped.begin(),
                    [](unsigned char c) { return (char) std::tolower(c); });
    return stripped;
}

// CLIP's own regex (distinct from GPT-2's): single digits, not digit runs, and no "leading space
// attaches to the next run" rule -- whitespace is simply not matched by any alternative, so it's
// dropped, matching CLIP's byte-level-but-not-space-preserving tokenization.
static std::vector<std::string> clip_split(const std::string & text) {
    static const std::regex pat(R"('s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                                 std::regex::icase);
    std::vector<std::string> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pat); it != std::sregex_iterator(); ++it) {
        out.push_back(it->str());
    }
    return out;
}

struct clip_bpe_data {
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<std::string, int> bpe_ranks;
    std::unordered_map<std::string, int32_t> token_to_id;
};

static const clip_bpe_data & get_bpe_data() {
    static clip_bpe_data data = [] {
        clip_bpe_data d;
        std::array<uint8_t, 256> bs_order;
        build_byte_tables(bs_order, d.byte_encoder);

        std::vector<std::string> vocab;
        vocab.reserve(49408);
        for (int i = 0; i < 256; i++) vocab.push_back(d.byte_encoder[bs_order[i]]);
        for (int i = 0; i < 256; i++) vocab.push_back(d.byte_encoder[bs_order[i]] + "</w>");

        std::istringstream iss(CLIP_MERGES_DATA);
        std::string line;
        // Skip up to and including the "#version: 0.2" header line. The embedded raw-string
        // literal picks up a blank line right after its opening delimiter before that header, so
        // skip blank lines first rather than assuming the very first getline is the header.
        do { std::getline(iss, line); } while (iss && line.empty());
        int rank = 0;
        while (std::getline(iss, line)) {
            if (line.empty()) continue; // a real merge token can legitimately start with '#', so
                                         // only blank lines are safe to skip unconditionally here
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string left  = line.substr(0, sp);
            std::string right = line.substr(sp + 1);
            if (!right.empty() && right.back() == '\r') right.pop_back();
            d.bpe_ranks[left + "\x01" + right] = rank++;
            vocab.push_back(left + right);
        }
        vocab.push_back("<|startoftext|>");
        vocab.push_back("<|endoftext|>");

        for (size_t i = 0; i < vocab.size(); i++) {
            d.token_to_id[vocab[i]] = (int32_t) i;
        }
        return d;
    }();
    return data;
}

clip_vocab load_clip_vocab() {
    clip_vocab v;
    v.token_to_id = get_bpe_data().token_to_id;
    return v;
}

std::vector<int32_t> clip_vocab::tokenize(const std::string & text, int n_ctx) const {
    const clip_bpe_data & bpe = get_bpe_data();

    std::vector<int32_t> ids;
    ids.push_back(49406); // <|startoftext|>

    std::string normalized = whitespace_clean_lower(text);
    for (const std::string & word : clip_split(normalized)) {
        std::vector<std::string> symbols;
        symbols.reserve(word.size());
        for (unsigned char b : word) {
            symbols.push_back(bpe.byte_encoder[b]);
        }
        symbols.back() += "</w>";

        for (const std::string & piece : bpe_merge(symbols, bpe.bpe_ranks)) {
            auto it = bpe.token_to_id.find(piece);
            if (it != bpe.token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                fprintf(stderr, "warning: no CLIP vocab id for BPE piece '%s'\n", piece.c_str());
            }
        }
        if ((int) ids.size() >= n_ctx - 1) break; // leave room for EOS below
    }

    ids.resize(std::min((size_t) ids.size(), (size_t)(n_ctx - 1)));
    ids.push_back(49407); // <|endoftext|>
    ids.resize(n_ctx, 49407); // pad with EOS/PAD (same id) up to the fixed context length
    return ids;
}

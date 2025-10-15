#include "kseq/kseq.h"
#include "common.h"

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
    matches.clear();

    for (const auto &sample : samples) {
        const std::string &sample_seq = sample.seq;
        int integrity_hash = 0;
        for (char c : sample.qual) {
            integrity_hash += c - 33;
        }
        integrity_hash %= 97;
        for (const auto &sig : signatures) {
            const std::string &sig_seq = sig.seq;

            for (size_t i = 0; i <= sample_seq.length() - sig_seq.length(); i++) {
                bool found = true;
                for (size_t j = 0; j < sig_seq.length(); j++) {
                    if (sample_seq[i + j] != sig_seq[j]) {
                        found = false;
                        break;
                    }
                }

                if (found) {
                    MatchResult match;
                    match.sample_name = sample.name;
                    match.signature_name = sig.name;
                    match.integrity_hash = integrity_hash;
                    double score_sum = 0.0;
                    for (size_t j = 0; j < sig_seq.length(); ++j) {
                        int q = static_cast<int>(sample.qual[i + j]) - 33;
                        score_sum += q;
                    }
                    match.match_score = score_sum / static_cast<double>(sig_seq.length());

                    matches.push_back(match);
                    break;
                }
            }
        }
    }
}

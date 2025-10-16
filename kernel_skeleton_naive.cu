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
            size_t sig_len = sig_seq.length();
            size_t sam_len = sample_seq.length();

            double best_match_score = -1e9;
            size_t best_match_pos = sam_len;
            for (size_t i = 0; i + sig_len <= sam_len; i++) {
                bool found = true;
                for (size_t j = 0; j < sig_len; j++) {
                    if (sample_seq[i + j] != 'N' && sample_seq[i + j] != sig_seq[j]) {
                        found = false;
                        break;
                    }
                }

                if (found) {
                    double score_sum = 0.0;
                    for (size_t j = 0; j < sig_len; ++j) {
                        int q = sample.qual[i + j] - 33;
                        score_sum += q;
                    }
                    double match_score = score_sum / static_cast<double>(sig_len);

                    if (match_score > best_match_score) {
                        best_match_score = match_score;
                        best_match_pos = i;
                    }
                }
            }

            if (best_match_pos != sam_len) {
                MatchResult match;
                match.sample_name = sample.name;
                match.signature_name = sig.name;
                match.integrity_hash = integrity_hash;
                match.match_score = best_match_score;
                matches.push_back(match);
            }
        }
    }
}
